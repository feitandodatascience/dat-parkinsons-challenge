# External assets register

The competition allows external data and pretrained models **only** if you hold all rights,
licences and permissions, "without the need for additional approvals from any third party",
and can prove it on request. Winners must also document what their solution needs to build and
run.

**Log every external asset here, at the moment you first use it.** Reconstructing provenance
during the final week is how people discover on day 15 that a set of weights is
non-commercial-only and cannot be used.

## Register

| Asset | Source URL | Version / commit | Licence | Commercial use OK? | Where used | Date added |
|---|---|---|---|---|---|---|
| _(none yet)_ | | | | | | |

## Checklist before adding anything

- [ ] Licence identified and **read**, not assumed from the repo's popularity
- [ ] Licence permits use in a competition submission that will be released under MIT
- [ ] No additional third-party approval needed
- [ ] Asset can be **vendored into submission.zip** — the runtime has no network
- [ ] Total zip size still reasonable
- [ ] Row added to the table above

## Notes on likely candidates

- **MONAI / MedicalNet pretrained 3D weights** — check the specific weights' licence. MONAI
  itself is Apache 2.0, but bundled model weights sometimes carry separate, more restrictive
  terms.
- **`timm` ImageNet backbones** — mostly Apache 2.0 / MIT, but confirm per-model. Some
  weights (certain EfficientNet and ViT variants) have their own conditions.
## Ruled out

- **PPMI data** (<https://www.ppmi-info.org>) — **NOT usable in this competition. Settled, not
  a judgement call.** The organizers ruled on it directly
  ([forum thread](https://community.drivendata.org/t/is-ppmi-dua-gated-acceptable-as-external-data-for-prize-eligibility/11472/2)):
  the PPMI Data Use Agreement permits use only for "scientific investigation, teaching, or the
  planning of clinical research studies". Competition use is none of those, and winning
  solutions must be openly licensed for commercial use — which a DUA-gated dataset cannot
  support. **Models trained on PPMI data are not eligible for prizes.**

  This applies transitively: it rules out PPMI-derived *pretrained weights* just as much as the
  images, since the weights inherit the DUA's purpose restriction. Before using any public DaT
  checkpoint, check what it was trained on — a great many published DaT models are PPMI-based
  and are therefore out of scope here.

  Reading PPMI's *papers and documentation* is fine and still worthwhile — that is literature,
  not Data. It is the images and anything derived from them that are excluded.

  The general lesson worth carrying: "publicly available" is not the same as "licensed for
  this use". A DUA that gates access almost always also constrains purpose.
