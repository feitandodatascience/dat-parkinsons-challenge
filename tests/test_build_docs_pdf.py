"""Tests for the docs -> PDF build.

The link-rewriting and glyph tests always run. The tests that compile a PDF are skipped when
pandoc or a LaTeX engine is absent, so this suite stays green on a machine that only ever reads
the Markdown.
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "build_docs_pdf.py"

_spec = importlib.util.spec_from_file_location("build_docs_pdf", SCRIPT)
assert _spec and _spec.loader
build_docs_pdf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_docs_pdf)

# Pandoc is installed per-user on Windows and may not be on PATH in a bare shell.
_PANDOC_CANDIDATES = [
    shutil.which("pandoc"),
    str(Path.home() / "AppData" / "Local" / "Pandoc" / "pandoc.exe"),
]
PANDOC = next((p for p in _PANDOC_CANDIDATES if p and Path(p).exists()), None)
ENGINE = shutil.which("lualatex")

needs_pandoc = pytest.mark.skipif(
    PANDOC is None or ENGINE is None, reason="pandoc and/or lualatex not available"
)


class TestPreprocess:
    def test_doc_cross_references_become_pdf_links(self) -> None:
        out = build_docs_pdf.preprocess("see [the primer](01_domain_primer_dat_scans.md) now")
        assert "(01_domain_primer_dat_scans.pdf)" in out

    def test_relative_doc_links_are_retargeted(self) -> None:
        out = build_docs_pdf.preprocess("[brief](docs/00_competition_brief.md)")
        assert "(00_competition_brief.pdf)" in out

    def test_source_links_become_inline_code(self) -> None:
        """A hyperlink into the source tree is dead in a PDF; show the path instead."""
        out = build_docs_pdf.preprocess("see [preprocess.py](../src/datpd/data/preprocess.py)")
        assert "`src/datpd/data/preprocess.py`" in out
        assert "](" not in out

    def test_external_links_are_untouched(self) -> None:
        md = "[runtime](https://github.com/drivendataorg/competition-sfmn-parkinsons-runtime)"
        assert build_docs_pdf.preprocess(md) == md

    def test_anchor_links_are_untouched(self) -> None:
        assert build_docs_pdf.preprocess("[top](#overview)") == "[top](#overview)"

    def test_unsupported_glyphs_are_substituted(self) -> None:
        out = build_docs_pdf.preprocess("✅ yes ❌ no ⚠️ careful ☐ todo")
        assert "[YES]" in out and "[NO]" in out and "[!]" in out and "[ ]" in out
        assert "✅" not in out and "❌" not in out

    def test_box_drawing_is_preserved(self) -> None:
        """Consolas covers these; substituting ASCII would wreck the diagrams."""
        diagram = "┌──────┐\n│ box  │\n└──────┘\n█▒·"
        assert build_docs_pdf.preprocess(diagram) == diagram


class TestRealDocs:
    @pytest.mark.parametrize("stem", [s for s, _ in build_docs_pdf.READING_DOCS])
    def test_no_dead_relative_links(self, stem: str) -> None:
        import re

        out = build_docs_pdf.preprocess(
            (REPO_ROOT / "docs" / f"{stem}.md").read_text(encoding="utf-8")
        )
        links = re.findall(r"\]\(([^)]+)\)", out)
        dead = [l for l in links if not l.startswith(("http", "#")) and not l.endswith(".pdf")]
        assert not dead, f"{stem} has links that will not resolve in a PDF: {dead}"

    @pytest.mark.parametrize("stem", [s for s, _ in build_docs_pdf.READING_DOCS])
    def test_every_reading_doc_exists(self, stem: str) -> None:
        assert (REPO_ROOT / "docs" / f"{stem}.md").exists()


@needs_pandoc
class TestRenderedOutput:
    """Compile a small document and check the PDF text round-trips."""

    @staticmethod
    def _render(tmp_path: Path, markdown: str) -> str:
        pypdf = pytest.importorskip("pypdf")

        md = tmp_path / "t.md"
        md.write_text(markdown, encoding="utf-8")
        pdf = tmp_path / "t.pdf"

        result = subprocess.run(
            [PANDOC, str(md), "-o", str(pdf), "--pdf-engine=lualatex", "--from", "gfm",
             "-V", "mainfont=Georgia", "-V", "monofont=Consolas"],
            capture_output=True, text=True, cwd=tmp_path,
        )
        assert result.returncode == 0, result.stderr[-800:]

        reader = pypdf.PdfReader(str(pdf))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    def test_shell_commands_stay_copy_pasteable(self, tmp_path: Path) -> None:
        """The reason the build uses LuaLaTeX rather than XeLaTeX.

        XeLaTeX maps Consolas' hyphen to U+2010 HYPHEN in the PDF's ToUnicode table, so a
        command copied out of the PDF ("py -3.12") silently becomes "py ‐3.12" and fails in a
        shell. These docs exist to be copied from, so this must round-trip as ASCII.
        """
        text = self._render(tmp_path, "Run `py -3.12 -m venv .venv` then `--config x.yaml`.")
        assert "‐" not in text, "hyphen became U+2010; commands will not paste correctly"
        assert "py -3.12 -m venv .venv" in text
        assert "--config x.yaml" in text

    def test_fenced_code_block_round_trips(self, tmp_path: Path) -> None:
        text = self._render(
            tmp_path,
            "```bash\npython scripts/03_train.py --config conf/baseline.yaml\n```\n",
        )
        assert "‐" not in text
        assert "--config conf/baseline.yaml" in text

    def test_pipe_table_renders(self, tmp_path: Path) -> None:
        text = self._render(tmp_path, "| a | b |\n|---|---|\n| 1 | 2 |\n")
        assert "a" in text and "b" in text and "1" in text and "2" in text


@needs_pandoc
def test_full_build_produces_readable_pdfs(tmp_path: Path) -> None:
    """End-to-end: build every doc and assert the content actually landed in the PDF."""
    pypdf = pytest.importorskip("pypdf")

    env_path = str(Path(PANDOC).parent)
    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        capture_output=True, text=True, cwd=REPO_ROOT,
        env={**__import__("os").environ, "PATH": env_path + ";" + __import__("os").environ["PATH"],
             "PYTHONIOENCODING": "utf-8"},
    )
    assert result.returncode == 0, result.stdout[-2000:]

    for stem, _ in build_docs_pdf.READING_DOCS:
        pdf = REPO_ROOT / "docs" / "pdf" / f"{stem}.pdf"
        assert pdf.exists(), f"{stem}.pdf was not produced"
        reader = pypdf.PdfReader(str(pdf))
        assert len(reader.pages) >= 3, f"{stem}.pdf looks truncated"
        text = "\n".join(p.extract_text() or "" for p in reader.pages)
        assert len(text) > 5000, f"{stem}.pdf has suspiciously little text"
        assert "‐" not in text, f"{stem}.pdf has non-ASCII hyphens"
