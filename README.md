# Automatic Annotation of Human Brain Regions

<p align="left">
  <img src="https://img.shields.io/badge/Task-Region%20Segmentation-1565C0?style=flat-square">
  <img src="https://img.shields.io/badge/Scale-Whole%20Brain-7B1FA2?style=flat-square">
  <img src="https://img.shields.io/badge/Output-GeoJSON%20%7C%20OpenSeadragon-009688?style=flat-square">
  <img src="https://img.shields.io/badge/code-private%20(institutional%20IP)-555?style=flat-square">
</p>

> Built at the **Sudha Gopalakrishnan Brain Centre, IIT Madras**, on serial whole-brain histology. Documents method + results; source held under institutional IP.

---

## The problem

A single brain in the archive is reconstructed from **~10,000 serial sections**. Anatomical regions (cortical plate, white matter, sub-plate, ventricular zones, etc.) must be delineated on every section for atlas building and 3D reconstruction. Doing this by hand, per section, per brain, simply does not scale — and manual boundaries drift between annotators.

## The approach

A model that proposes region boundaries automatically and emits them as editable annotations a neuroanatomist can correct, not redraw.

![Annotation pipeline](./assets/architecture.svg)

1. **Section input** — a registered whole-slide section (Nissl / IHC).
2. **Region model** — a segmentation model predicts anatomical region masks across the section.
3. **Vectorization** — raster masks are converted to clean polygon boundaries.
4. **GeoJSON export** — boundaries are written as GeoJSON, the interoperable standard for slide annotation, so they load into existing review tools.
5. **Human-in-the-loop review** — annotations render in an **OpenSeadragon** deep-zoom viewer where an expert nudges boundaries instead of drawing from scratch.

This connects directly to the centre's broader **3D fetal brain reconstruction** effort, where consistent per-section regions are the input to volumetric assembly.

## Key results

- Produces editable region annotations across whole sections at archive resolution (0.5 µm/pixel).
- Cuts annotation from *draw-everything* to *review-and-adjust* — the throughput unlock that makes whole-brain coverage realistic.

> _Add a side-by-side of predicted vs. expert-corrected regions, and a multi-region overlay, here._
>
> `![Region overlay](./assets/region-overlay.png)`

## Tech stack

`PyTorch` · `Segmentation models` · `shapely / GeoJSON` · `OpenSeadragon` · `FastAPI`

## Why it matters

This is the annotation-infrastructure layer beneath an entire brain-mapping program. It pairs a real ML model with a deployment that respects how domain experts actually work — proposals they refine — which is exactly how clinical/scientific AI gets adopted.

---

<sub>Code is private under IIT Madras / SGBC institutional agreements. Walkthrough available on request.</sub>
