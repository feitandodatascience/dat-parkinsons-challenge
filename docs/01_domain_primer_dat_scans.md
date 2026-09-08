# 01 — Domain primer: what a DaT scan actually is

Written for someone starting from zero on nuclear medicine. Target: after ~40 minutes you
should understand what the images show, what "abnormal" means, why ~1 in 5 cases is hard, and
which of those facts should change your model. Nothing here requires prior biology.

---

## 1. The clinical problem

Parkinsonian syndromes — Parkinson's disease and its relatives — produce tremor, rigidity, and
slowness of movement. The trouble is that **several very different conditions look similar at
the bedside**, and the treatment and prognosis differ sharply:

| Looks parkinsonian, IS neurodegenerative | Looks parkinsonian, is NOT |
|---|---|
| Parkinson's disease (PD) | Essential tremor |
| Multiple system atrophy (MSA) | Drug-induced parkinsonism (e.g. from antipsychotics) |
| Progressive supranuclear palsy (PSP) | Vascular parkinsonism |
| Corticobasal degeneration (CBD) | Functional / psychogenic movement disorder |
| Dementia with Lewy bodies (DLB) | |

Getting this wrong means years of the wrong drug, or a wrong prognosis. Clinical examination
alone misclassifies a meaningful fraction of cases, especially early, when signs are subtle.

**This is where the DaT scan earns its place.** It answers one specific question well:
*is the dopamine system actually degenerating, or not?*

---

## 2. The biology, in five sentences

1. A set of neurons in the **substantia nigra** (midbrain) sends dopamine-releasing fibres up
   into a structure called the **striatum**.
2. On the tips of those fibres sits a protein, the **dopamine transporter (DAT)**, whose job is
   to hoover dopamine back out of the synapse.
3. In degenerative parkinsonism those neurons die, so the fibres retract, so **there is less
   DAT in the striatum**.
4. The DaT scan injects a radioactive tracer that **sticks to DAT**, so the amount of signal in
   the striatum is a proxy for how many dopamine terminals survive.
5. Conditions in the right-hand column above leave the terminals intact, so their scans look
   **normal** — which is exactly the discrimination the scan provides.

Important limit, worth internalising: **the scan says "degenerating / not degenerating". It
does not say "PD vs MSA vs PSP"** — all of those are abnormal. Our binary label is
`is_pathologic`, which matches what the scan can actually support.

---

## 3. How the image is made

- **Tracer:** ioflupane labelled with iodine-123, commonly written **I-123 FP-CIT**. The trade
  name **DaTscan** is where the competition name comes from.
- **Prep:** the patient takes potassium iodide to stop the thyroid taking up free iodine.
- **Wait:** imaging happens **3–6 hours** after injection, once the tracer has bound and
  unbound tracer has cleared.
- **Acquisition:** **SPECT** — Single Photon Emission Computed Tomography. A gamma camera
  rotates around the head counting individual photons.
- **Reconstruction:** projections are turned into a 3D volume (OSEM iterative reconstruction,
  or older filtered back-projection), possibly with attenuation and scatter correction.

### What this means for you as a modeller

| Property | Value | Consequence |
|---|---|---|
| Spatial resolution | ~8–12 mm FWHM | Very blurry. Fine texture is mostly noise; do not expect edges. |
| Counting statistics | Low | Poisson noise is substantial. Aggressive denoising can erase real signal. |
| Voxel units | Raw counts, uint16 | **Not calibrated between scanners.** Absolute intensity is meaningless across centres. |
| Reconstruction | Varies by centre | Different smoothing and correction produce different textures for identical anatomy. |

That third row is why per-scan intensity normalisation is not optional. It is also why
clinicians read **ratios**, not absolute values (§5).

The reconstruction row is why this dataset is hard and why the competition exists: a model can
easily learn *which hospital took the scan* instead of *whether the patient is ill*.

---

## 4. What normal and abnormal look like

The striatum has two parts, and their names appear constantly in the literature:

- **Caudate nucleus** — the anterior (front) part.
- **Putamen** — the posterior (back) part. Subdivided into anterior and posterior putamen.

On a normal scan, each side of the brain shows a bright, symmetric **comma** or **crescent**
shape: the caudate is the fat head of the comma, the putamen its tail sweeping backwards.

Degeneration follows a stereotyped spatial pattern:

```
                  ANTERIOR (front of head)
                          |
     NORMAL               |            ABNORMAL
                          |
   caudate  ████          |          caudate  ███       <- goes last, partially spared
      ▲     ████          |             ▲     ███
      |     ███           |             |     ██
   putamen  ███           |          putamen  ▒          <- goes FIRST, worst affected
      ▼     ██            |             ▼     ·
                          |
   "comma" / "crescent"   |          "period" / "full stop"
                  POSTERIOR (back of head)
```

Three patterns to know by name, because they are the actual signal:

1. **Posterior-to-anterior gradient.** The posterior putamen loses signal first, then anterior
   putamen, then caudate. The comma loses its tail and becomes a dot.
2. **Asymmetry.** Loss is typically worse on one side — the side opposite the patient's more
   symptomatic limb. Bilateral symmetric loss is more suggestive of atypical syndromes.
3. **Background rises relative to striatum.** As specific binding falls, the striatum stands
   out less against surrounding tissue, and the image looks flatter overall.

> **Modelling consequence (this one is important).**
> Left-right flipping is a reflexive augmentation in natural-image CV. Here it may destroy
> signal, because **asymmetry is diagnostic** and laterality carries meaning. It is disabled by
> default in [conf/baseline.yaml](../conf/baseline.yaml). Run one controlled experiment before
> turning it on, and judge it on OOF log loss.

---

## 5. How clinicians actually quantify it: the SBR

Radiologists do not read raw counts. They compute a **specific binding ratio**:

```
SBR = (mean counts in striatal region − mean counts in reference region)
      ─────────────────────────────────────────────────────────────────
                    mean counts in reference region
```

The **reference region** is a part of the brain with essentially no DAT — usually the occipital
cortex, sometimes whole brain or cerebellum. Its signal is "non-specific" binding, i.e. the
background level of tracer that is simply present rather than bound to anything.

Dividing by the reference cancels out scanner gain, injected dose, and patient size. That is
precisely why `normalize_intensity(method="reference_region")` in
[preprocess.py](../src/datpd/data/preprocess.py) is the default: it is a cheap approximation of
the quantity clinicians trust, and it is scale-invariant by construction (there is a test
asserting exactly that).

Related quantities you will meet in papers: **putamen SBR**, **caudate SBR**, the
**putamen/caudate ratio** (captures the gradient), and the **asymmetry index**
`|left − right| / (left + right)`. All four are strong, interpretable, hand-engineered features.
If you register scans to a common template, computing them explicitly is a credible non-deep
baseline — and a useful sanity check on whatever your CNN learns.

---

## 6. Why ~1 in 5 cases is hard

The organizers state that roughly one in five DaT scans is difficult to interpret. Knowing the
reasons tells you where your log loss will be spent:

- **Early disease.** Signal loss is gradual. A scan taken near the threshold is genuinely
  borderline, and two experts may disagree.
- **Age.** DAT density falls naturally with age, roughly 5–8% per decade. A mildly reduced scan
  in an 80-year-old may be normal for their age; the same image in a 45-year-old is not. This
  is real, and it is why the organizers float age/sex prediction as follow-up work.
- **SWEDD** — "Scans Without Evidence of Dopaminergic Deficit": patients with a clinical
  parkinsonism diagnosis but a normal scan.
- **Scanner and reconstruction differences.** A different reconstruction filter can shift
  apparent contrast enough to move a borderline case.
- **Motion and attenuation artefacts.**

> **Modelling consequence.** Ambiguity is irreducible, and **log loss rewards you for admitting
> it**. A model that outputs 0.5 on genuinely borderline scans and 0.02 / 0.98 on clear ones
> beats a model that is confident everywhere. Do not chase accuracy; chase honest probabilities.
> This is the single biggest mindset shift from MNIST, where labels are unambiguous.

---

## 7. Anatomy cheat-sheet

| Term | Plain meaning |
|---|---|
| Striatum | The target structure. Caudate + putamen. Where the tracer concentrates. |
| Caudate nucleus | Front part of striatum. Head of the "comma". Spared longest. |
| Putamen | Back part. Tail of the "comma". Degenerates first, posterior end worst. |
| Substantia nigra | Midbrain source of the dopamine neurons. Too small to see on SPECT. |
| Occipital cortex | Back of the brain. Used as the DAT-free reference region. |
| Ipsilateral / contralateral | Same side / opposite side. Brain loss is contralateral to symptoms. |
| Anterior / posterior | Front / back. |
| Superior / inferior | Up / down. |
| Left / right | In radiological convention images may be L-R mirrored — check orientation codes, do not assume. |
| SBR | Specific binding ratio (§5). |
| SPECT | The imaging modality. Photon counting, blurry, noisy. |
| I-123 FP-CIT / ioflupane | The tracer. |
| SWEDD | Clinically parkinsonian, normal scan. |
| RAS / LPS | Axis orientation codes. `nibabel.as_closest_canonical` normalises to RAS. |

---

## 8. The five things to carry into your modelling

1. **Resample to millimetres.** Physical scale must be consistent, or convolutions are
   incoherent across centres. ([preprocess.py](../src/datpd/data/preprocess.py))
2. **Normalise per scan, ratio-style.** Absolute counts are scanner-dependent and meaningless
   across sites; ratios are what the biology and the clinicians both use.
3. **Do not flip left-right by default.** Asymmetry is signal, not nuisance.
4. **Centre is a confounder.** Ten hospitals, different scanners. Use grouped CV
   ([splits.py](../src/datpd/data/splits.py)) or you will fool yourself.
5. **Calibrate.** One in five cases is genuinely ambiguous, and log loss pays you to say so.
   ([calibrate.py](../src/datpd/calibrate.py))

---

## 9. Where to read more

Roughly in order of usefulness per minute spent. None is required before you start coding.

- **Radiopaedia — "DaT scan"** and **"Dopamine transporter imaging"**: short, image-led, ideal
  for building visual intuition of the comma-to-dot progression. Start here.
- **EANM/SNMMI practice guideline for dopamine transporter imaging** (search: *EANM DaT SPECT
  guideline*): how the scan is acquired and read, including reference-region conventions.
- **PPMI (Parkinson's Progression Markers Initiative)**, <https://www.ppmi-info.org>: the large
  public DaT dataset. Its papers and DICOM conventions are the background literature for most
  published DaT deep learning, and reading them is worthwhile.
  **The data itself cannot be used in this competition** — the organizers have ruled that its
  Data Use Agreement does not cover competition use, so models trained on it are not eligible
  for prizes. That extends to PPMI-derived pretrained weights. See
  [EXTERNAL_ASSETS.md](EXTERNAL_ASSETS.md) for the ruling and its reasoning. Read the
  literature; do not download the images.
- Search **"deep learning DaT SPECT classification"** on PubMed for prior art. Expect small,
  single-centre studies reporting high accuracy — treat those numbers as optimistic; the whole
  point of this competition is that multicentre generalisation is harder.
