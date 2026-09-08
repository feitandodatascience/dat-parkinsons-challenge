#!/usr/bin/env python3
"""Render the reading docs to PDF for offline study.

The Markdown files stay the source of truth (Claude reads those, and they cross-link into the
code). This script produces a parallel set of PDFs in `docs/pdf/` for comfortable reading away
from the editor.

Three preprocessing steps happen before pandoc sees the text, each fixing something that
otherwise looks wrong in print:

1. **Doc-to-doc links are retargeted to `.pdf`**, so cross-references still work in a PDF
    reader.
2. **Links into the source tree become inline code.** A clickable link to `../src/datpd/train.py`
    is dead in a PDF; showing the path in monospace is what a reader actually needs.
3. **Glyphs the fonts lack are substituted.** XeLaTeX drops an unmapped character
    silently, so a tick or a warning sign becomes an invisible gap. Box-drawing characters
    are left alone -- Consolas has them, and ASCII substitutes would spoil the diagrams.
    Run with `--check-glyphs` to confirm nothing was dropped.

Both the English docs and their Vietnamese `_vn` translations are built. The two sets need
different body fonts (Georgia has no Vietnamese coverage); see LANGS.

Usage:
    python scripts/build_docs_pdf.py
    python scripts/build_docs_pdf.py --docs 01 03      # only some
    python scripts/build_docs_pdf.py --lang vn         # only the translations
    python scripts/build_docs_pdf.py --check-glyphs    # verify nothing is silently dropped
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = REPO_ROOT / "docs"
OUT_DIR = DOCS_DIR / "pdf"

# The docs meant to be read start to finish, in reading order.
READING_DOCS = [
    ("01_domain_primer_dat_scans", "Domain Primer: DaT Scans"),
    ("02_cv_for_3d_medical_imaging", "From MNIST to 3D Medical Imaging"),
    ("00_competition_brief", "Competition Brief"),
    ("03_plan", "Working Plan"),
    ("04_working_with_claude_code", "Working with Claude Code"),
]

# Vietnamese translations, same reading order, `_vn` suffix. They are a parallel set rather
# than a variant of the same build because they need a different body font (see LANGS).
READING_DOCS_VN = [
    ("01_domain_primer_dat_scans_vn", "Nhập môn lĩnh vực: ảnh DaT scan"),
    ("02_cv_for_3d_medical_imaging_vn", "Từ MNIST tới ảnh y tế 3D"),
    ("00_competition_brief_vn", "Tóm tắt cuộc thi"),
    ("03_plan_vn", "Kế hoạch làm việc"),
    ("04_working_with_claude_code_vn", "Làm việc với Claude Code"),
]

# Per-language build settings.
#
# The body font is the whole reason these are separate builds. Georgia has no coverage of the
# Vietnamese Latin Extended Additional block (U+1EA0-U+1EF9) or of U+01A0/U+01AF (o-horn,
# u-horn), so every tone-marked vowel would be dropped silently -- exactly the failure mode
# `--check-glyphs` exists to catch. Cambria covers them and is the closest match to Georgia's
# weight and colour on the page. Consolas is unchanged: it covers Vietnamese *and* the
# box-drawing characters, so the ASCII diagrams survive the switch intact.
#
# KNOWN LIMITATION, Vietnamese only: the tone marks render correctly but do not survive text
# extraction -- copying "Nhập môn" out of the PDF yields "Nha p mo n", and in-PDF search for a
# tone-marked word fails. The ToUnicode map luaotfload writes for the Latin Extended Additional
# range (U+1EA0-U+1EF9) does not round-trip, and this is not engine-specific (XeLaTeX is the
# same) nor fixable among the fonts on a stock Windows install -- Times New Roman, Arial and
# Calibri are all strictly worse, dropping the base letters too. A font with a clean Unicode
# cmap (Noto Serif, DejaVu Serif, Charis SIL) would fix it but none is installed. ASCII content
# -- code, commands, English terms -- extracts correctly, so copy-pasteable commands, the
# reason this script uses LuaLaTeX at all, still work. The Markdown sources stay fully
# searchable, so reading is unaffected; only copy/search of Vietnamese prose is degraded.
LANGS = {
    "en": {
        "docs": READING_DOCS,
        "mainfont": "Georgia",
        "subtitle": "DaT Parkinson's Challenge",
    },
    "vn": {
        "docs": READING_DOCS_VN,
        "mainfont": "Cambria",
        "subtitle": "DaT Parkinson's Challenge",
    },
}

# Glyphs with no coverage in the body font (Georgia) or the mono font (Consolas). Box-drawing
# (U+2500 block) and block elements (U+2588, U+2592) are deliberately NOT listed: Consolas
# renders them, and substituting ASCII would degrade the ASCII diagrams for no reason.
# `--check-glyphs` verifies this empirically against the XeLaTeX log rather than trusting it.
GLYPH_SUBSTITUTIONS = {
    "✅": "[YES]",
    "❌": "[NO]",
    "⚠️": "[!]",
    "⚠": "[!]",
    "☐": "[ ]",
    "☑": "[x]",
    "−": "-",      # U+2212 minus sign
    "≤": "<=",
    "≥": ">=",
    # U+2192 is absent from Georgia, so every prose arrow ("Claude writes it -> you run it")
    # was being dropped from the English PDFs. Cambria has it, but substitute for both so the
    # two language sets read identically. Only safe while the arrow stays out of code fences:
    # it is the one substitution here that changes width (1 char -> 2).
    "→": "->",
    # U+25B6/U+25C0 are absent from Consolas, which took the arrowheads off the workflow
    # diagram in docs/04. ASCII angle brackets are exactly one column wide, so the box
    # alignment survives -- unlike the substitution above, these are width-preserving.
    "▶": ">",
    "◀": "<",
}

DOC_STEMS = (
    {stem for stem, _ in READING_DOCS}
    | {stem for stem, _ in READING_DOCS_VN}
    | {"EXTERNAL_ASSETS"}
)


def preprocess(text: str) -> str:
    """Make the Markdown safe and sensible for PDF output."""

    def fix_link(match: re.Match[str]) -> str:
        label, target = match.group(1), match.group(2)
        if target.startswith(("http://", "https://", "#", "mailto:")):
            return match.group(0)

        stem = Path(target).stem
        # A cross-reference to another reading doc: point at its PDF.
        if stem in DOC_STEMS and target.endswith(".md"):
            return f"[{label}]({stem}.pdf)"
        # A link into the source tree or elsewhere in the repo: dead in a PDF, so render the
        # path as code instead of a broken hyperlink.
        clean = target.lstrip("./")
        while clean.startswith("../"):
            clean = clean[3:]
        return f"{label} (`{clean}`)"

    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", fix_link, text)

    for glyph, replacement in GLYPH_SUBSTITUTIONS.items():
        text = text.replace(glyph, replacement)

    return text


def build(stem: str, title: str, engine: str, lang: str = "en") -> bool:
    settings = LANGS[lang]
    src = DOCS_DIR / f"{stem}.md"
    if not src.exists():
        print(f"  [skip] {stem}.md not found")
        return False

    processed = preprocess(src.read_text(encoding="utf-8"))

    with tempfile.TemporaryDirectory() as tmp:
        staged = Path(tmp) / f"{stem}.md"
        staged.write_text(processed, encoding="utf-8")

        out = OUT_DIR / f"{stem}.pdf"
        cmd = [
            "pandoc", str(staged),
            "-o", str(out),
            f"--pdf-engine={engine}",
            "--from", "gfm+pipe_tables+task_lists",
            "--toc", "--toc-depth=2",
            "--number-sections",
            "-V", f"title={title}",
            "-V", f"subtitle={settings['subtitle']}",
            "-V", "geometry:margin=2.2cm",
            "-V", "fontsize=11pt",
            "-V", "colorlinks=true",
            "-V", "linkcolor=RoyalBlue",
            "-V", "urlcolor=RoyalBlue",
            "-V", "toccolor=black",
        ]
        if engine in ("xelatex", "lualatex"):
            # Fonts present on a stock Windows install. Consolas is the important one: it
            # covers the box-drawing and block characters used in the ASCII diagrams, and a
            # proportional fallback would scramble their alignment.
            cmd += [
                "-V", f"mainfont={settings['mainfont']}",
                "-V", "sansfont=Segoe UI",
                "-V", "monofont=Consolas",
                "-V", "monofontoptions=Scale=0.85",
            ]

        result = subprocess.run(cmd, capture_output=True, text=True,
                                encoding="utf-8", errors="replace")
        if result.returncode != 0:
            print(f"  [FAIL] {stem}")
            print("    " + (result.stderr.strip().splitlines() or ["unknown error"])[-1][:400])
            return False

    print(f"  [ok]   {out.relative_to(REPO_ROOT)}  ({out.stat().st_size / 1024:.0f} KB)")
    return True


def check_glyphs(stem: str, engine: str, lang: str = "en") -> list[str]:
    """Compile via .tex and report characters XeLaTeX had no glyph for.

    XeLaTeX drops an uncovered character silently and only notes it in the log, so a missing
    tick or box-drawing character would otherwise become an invisible gap that nobody catches
    until they read the PDF. This turns that into an explicit list.
    """
    src = DOCS_DIR / f"{stem}.md"
    processed = preprocess(src.read_text(encoding="utf-8"))

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        staged = tmpdir / f"{stem}.md"
        staged.write_text(processed, encoding="utf-8")
        tex = tmpdir / f"{stem}.tex"

        subprocess.run(
            ["pandoc", str(staged), "-o", str(tex), "-s", "--from", "gfm+pipe_tables+task_lists",
             "-V", f"mainfont={LANGS[lang]['mainfont']}",
             "-V", "monofont=Consolas", "-V", "sansfont=Segoe UI"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", check=True,
        )
        # LuaLaTeX writes non-UTF-8 bytes into its console output, which the locale codec
        # (cp1252 on Windows) refuses. Decoding strictly here raised UnicodeDecodeError and
        # took down the very check meant to catch silent glyph drops.
        subprocess.run(
            [engine, "-interaction=nonstopmode", "-halt-on-error=false", tex.name],
            cwd=tmpdir, capture_output=True, text=True, encoding="utf-8", errors="replace",
        )

        log = tmpdir / f"{stem}.log"
        if not log.exists():
            return ["<no log produced>"]

        text = log.read_text(encoding="utf-8", errors="replace")
        return sorted(set(re.findall(r"Missing character: There is no (\S+)", text)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs", nargs="*", default=None,
                        help="numeric prefixes to build, e.g. --docs 01 03")
    # LuaLaTeX, not XeLaTeX, and the reason is copy-paste. Both render a hyphen correctly, but
    # XeLaTeX writes a ToUnicode map that reports Consolas' hyphen glyph as U+2010 HYPHEN
    # rather than U+002D HYPHEN-MINUS. Commands copied out of the PDF then fail in a shell
    # ("py -3.12" becomes "py ‐3.12"). These docs are full of commands meant to be copied,
    # so that matters. LuaLaTeX emits U+002D. Verified by tests/test_build_docs_pdf.py.
    parser.add_argument("--engine", default="lualatex",
                        choices=["lualatex", "xelatex", "pdflatex"])
    parser.add_argument("--check-glyphs", action="store_true",
                        help="report characters the fonts could not render, then exit")
    parser.add_argument("--lang", default="all", choices=["en", "vn", "all"],
                        help="which language set to build (default: all)")
    args = parser.parse_args()

    if shutil.which("pandoc") is None:
        print("pandoc not found. Install it with:  pip install pypandoc-binary")
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    langs = ["en", "vn"] if args.lang == "all" else [args.lang]
    # (lang, stem, title) triples, so each doc carries the font settings it needs.
    targets = [(lang, stem, title) for lang in langs for stem, title in LANGS[lang]["docs"]]
    if args.docs:
        wanted = set(args.docs)
        targets = [t for t in targets if t[1].split("_")[0] in wanted]

    if args.check_glyphs:
        print("Checking font coverage (characters the PDF engine would silently drop)")
        total = 0
        for lang, stem, _ in targets:
            missing = check_glyphs(stem, args.engine, lang)
            total += len(missing)
            # The missing characters are by definition ones the console may not encode either
            # (Windows stdout is cp1252 here), so report them as codepoints and let the glyph
            # itself be best-effort. Printing them raw crashes the check that exists to stop
            # silent drops -- which would be a particularly unhelpful failure.
            status = ", ".join(f"U+{ord(c):04X} {c!a}" for c in missing) if missing else "none"
            print(f"  {stem:36} {status}")
        print(f"\n{total} distinct missing glyph(s)")
        return 0 if total == 0 else 1

    print(f"Rendering {len(targets)} docs to {OUT_DIR.relative_to(REPO_ROOT)}/ via {args.engine}")
    ok = sum(build(stem, title, args.engine, lang) for lang, stem, title in targets)

    print(f"\n{ok}/{len(targets)} built")
    return 0 if ok == len(targets) else 1


if __name__ == "__main__":
    raise SystemExit(main())
