# Phase 3 research: vehicle-damage datasets and pretrained detectors

Status: desk research, October 2026. Nothing was downloaded, trained or benchmarked.
Every number below is quoted from a cited source; anything not found is marked
"not reported". Where a source could not be fetched, that is stated.

## TL;DR

1. **No usable off-the-shelf vehicle-damage YOLO weights were found.** The Hugging Face
   and Spaces candidates have no published class list, training data, metrics, or
   training config, or they keep their weights on external drives. Several have no
   model card at all. None are reproducible or auditable enough to produce stored
   `DamagePrediction`s for an insurance workflow.
2. **CarDD is the best-documented dataset for damage type plus localization.** It has
   4,000 images, 9,000+ instances, 6 classes, and box + mask + SOD labels. It is
   **non-commercial research and education only**, its image copyright stays with
   Flickr and Shutterstock, and access is by license request.
3. **Initial local integration test:** use the official Ultralytics **YOLO11n COCO**
   weights only as a *pipeline smoke test*, to check loading, coordinates, persistence
   and failure handling. They do **not** detect damage, and their output must never be
   presented as a damage assessment.
4. **Real damage detection needs a fine-tune**, for example YOLO11n/s on CarDD. That
   requires approval: the CarDD license form, Ultralytics AGPL-3.0 obligations, and the
   compute estimate in the last section.
5. **Severity and parts are not supported** by CarDD box/mask labels. They need a
   different labeled source or our own annotation, in a later phase.

## Datasets

| Dataset | Classes | Annotation | Size / splits | Access & license | Severity |
|---|---|---|---|---|---|
| **CarDD** (USTC, IEEE T-ITS 2023) [1][2][3] | dent, scratch, crack, glass shatter, lamp broken, tire flat | Instance masks; boxes derived from the masks ("tight-fitting… obtained from the pixel-level polygonal segmentation masks"); SOD maps | 4,000 images, >9,000 instances; train 2,816 / val 810 / test 374 | Licensing form, then a Google Drive link. "Non-commercial research and educational purposes" only. Images © Flickr/Shutterstock. Faces and plates mosaicked or removed. | The paper mentions a per-image "damage severity" spreadsheet; whether and how it is labeled is **not reported** in what was fetched. No per-instance severity. |
| **VehiDE** (KSE 2023; J. Inf. Telecom. 2025) [4][5] | 8 damage categories; **class names not verified** (paper page returned 403) | Classification, detection, instance segmentation and SOD reported | 13,945 images, >32,000 instances (secondary sources) | **License not verified** | Not reported |
| **Car Parts and Car Damages** (Humans in the Loop, Kaggle) [6] | Separate parts and damage subsets | Polygons (secondary source) | 1,812 images: 998 parts, 814 damages (secondary source) | Kaggle page did not render; **license not verified** | Not reported |
| **COCO Car Damage Detection** (small) [7] | Single class `damage`, plus a separate 5-class *parts* file | COCO | 59 train / 11 val / 8 test | Third-party mirror; license not verified | No |
| **Roboflow Universe car-damage projects** [8] | Varies by project. Some are generic, some mix parts with damage (e.g. "Scratch, Door, window, damage, wheel"), one uses French dent-per-panel labels | Boxes or polygons | Varies; mostly small | Many tagged CC BY 4.0 (pages returned 403 to direct fetch; seen in search listings only) | No |
| **Dent dataset** ("Small Dents, Big Impact", arXiv 2508.15431) [9] | Tiny dents | "annotated photos" | Not reported | No dataset link; article CC BY 4.0, dataset license not stated | No |

Quality and leakage notes:
- **CarDD:** the images are web-scraped, so near-duplicates across splits are possible.
  The paper's leakage handling was not checked. Our own evaluation should deduplicate by
  perceptual hash before trusting the published splits. Per-class instance counts are
  **not reported** in the paper text fetched, so class imbalance is unknown until the data
  is inspected.
- **Roboflow projects:** mixed taxonomies (parts and damage in one label space), unknown
  annotators, and frequent re-uploads of the same source images under different names.
  That makes cross-dataset leakage likely if several are combined.

## Pretrained detectors found

| Candidate | Arch | Classes | Training data | Metrics | License | Reproducible? |
|---|---|---|---|---|---|---|
| Ultralytics **YOLO11n** (COCO) [10][11] | YOLO11n, 2.6M params | 80 COCO classes (**no damage classes**) | COCO | mAP50-95 39.5 (COCO val, 640 px); CPU ONNX 56.1 ± 0.8 ms | AGPL-3.0 or Ultralytics Enterprise | Yes: official release |
| Ultralytics **YOLO11s** (COCO) [11] | 9.4M params | COCO | COCO | mAP50-95 47.0; CPU ONNX 90.0 ± 1.2 ms | AGPL-3.0 / Enterprise | Yes |
| `nezahatkorkmaz/car-damage-level-detection-yolov8` [12] | YOLOv8 | Light, Moderate, Severe (whole-damage severity, not type) | Unnamed dataset | Not reported | MIT tag on weights, but they come from an AGPL framework (see below) | No training config, data or metrics |
| `AItoolstack/car_damage_detection` [13][14] | YOLOv8 (detect/seg decided by the weights) | Not stated; classes live in the weights | Not stated | Not reported | Not stated | No. Card says weights are "downloaded at startup from public cloud links"; inference uses `conf=0.25`, `imgsz=640` |
| `Mariofm02/YOLOv8m_Car_Parts_Damaged` [15] | Presumably YOLOv8m (from the name only) | Not listed | Not described | None | Not shown | "No model card" |
| HF Spaces (e.g. `umerforsure/Car_Damage_Detection`) [16] | YOLOv8 `best.pt` | "scratch, dent, crack and more" (app text) | Not stated | Not reported | Not stated | Weights fetched from Google Drive |
| Dent YOLOv8m-t42 [9] | YOLOv8m variant | Dents | Bespoke | P 0.86, R 0.84, F1 0.85, mAP@0.5 0.60, mAP@0.5:0.95 0.20 | Paper CC BY 4.0 | No published checkpoint found |
| CarDD paper baseline [2] | DCN+ (ResNet-101; not YOLO) | CarDD 6 | CarDD | mask AP / box AP 57.0 / 60.6 (DCN: 52.5 / 54.3) | Not released as YOLO weights in sources fetched | Paper method |

Other non-YOLO community models exist on the Hub (DETR, RT-DETRv2, SegFormer, Florence-2,
PaliGemma fine-tunes) [17]. None were checked for class lists or licenses, so they are not
recommended without that check.

## What the labels support

- **Localization + damage type:** yes, with CarDD (6 types, boxes and masks), and probably
  VehiDE (8 types, unverified).
- **Damage / no-damage:** yes. Any box dataset gives image-level presence (no box means no
  damage). CarDD has no undamaged-car negatives in what was fetched, so false-positive rate
  on clean cars must be measured with our own negatives.
- **Severity:** **not supported** at instance level by any verified source. The one
  severity model found ([12]) has no documented data. Severity needs our own annotation
  protocol (e.g. minor / moderate / severe with written criteria and inter-annotator
  agreement) or a later VLM phase. It should not be inferred from box size.
- **Affected part (door, bumper…):** not in CarDD. Some Roboflow and Kaggle sets have parts
  as a separate label space. Linking damage to part needs a parts detector or annotation,
  which is later-phase work.

## Recommendations

**1. Weights for the initial local integration test:** official `yolo11n.pt`
(COCO), loaded through the `ultralytics` backend, used only to exercise the pipeline.
Config should label it plainly (e.g. `DETECTOR_MODEL_NAME=yolo11n-coco-smoke`) so stored
`DetectionRun`s cannot be mistaken for damage assessments.
- **Non-commercial student project:** acceptable under AGPL-3.0, provided the full source
  of the networked service stays published (AGPL §13 covers users interacting over a
  network). The repo is already public on GitHub.
- **Commercial or closed deployment:** requires the Ultralytics Enterprise License, or
  replacing the framework [10]. Ultralytics' maintainers state that both the code and the
  models are AGPL, *including* custom-trained models [18]. That position is disputed in the
  same thread and is **not legal advice**, so treat it as binding until counsel says
  otherwise.
- **Do not use the community `best.pt` files above.** They lack provenance, a class list,
  metrics and stable hosting, so `weights_sha256` would pin an artifact nobody can explain.

**2. Evaluation dataset:** the CarDD test split (374 images), plus a small in-house set of
our own captured photos. The in-house set covers clean cars (negatives), phone-camera
lighting and the capture angles our UI produces. CarDD's license keeps the result
research-only.

**3. Canonical taxonomy** (stored verbatim from the model in `damage_type`; map at read time
or in a later phase, without rewriting stored rows):

| Canonical | CarDD | Typical community labels | COCO smoke model |
|---|---|---|---|
| `dent` | dent | dent, bosse | — |
| `scratch` | scratch | scratch, paint damage | — |
| `crack` | crack | crack | — |
| `glass_damage` | glass shatter | broken glass, window damage | — |
| `lamp_damage` | lamp broken | broken lamp, headlight damage | — |
| `tire_damage` | tire flat | flat tire | — |
| `other_damage` | — | "damage" (generic), missing part | — |
| *(not damage)* | — | part labels (door, wheel…), severity labels | all 80 COCO classes |

Severity labels (Light/Moderate/Severe) are a separate axis and must not be mapped into
damage type.

**4. Evaluation plan**
- **Metrics:**
  - COCO-style mAP@0.5 and mAP@0.5:0.95, overall and **per class**.
  - Precision and recall at the *deployed* `confidence_threshold`; the operating point
    matters more than mAP for the product.
  - **Image-level damage recall:** fraction of damaged images with ≥1 correct detection;
    missed damage is the costly error for insurance.
  - False-positive rate on clean-car negatives.
  - Calibration: reliability diagram / ECE of confidence vs. precision.
  - Latency on the target CPU (M4) at 640 px.
- **Failure slices** (tag the in-house set): low light / night, strong reflections and
  glossy paint, close-up vs. full-car framing, dark vs. light paint, wet surfaces, motion
  blur, occlusion.
- **Hygiene:**
  - Freeze the test split before any tuning.
  - Deduplicate by perceptual hash across train/val/test and across merged sources.
  - Never tune the threshold on the test set; use val.
  - Record weights SHA-256, dataset version and config with every result. The existing
    `DetectionRun` columns already hold model, version, config and `weights_sha256`.
  - Report confidence intervals (bootstrap over images), because the CarDD test split
    is small.
- **Before downloading or fine-tuning** (needs approval):
  - Licensing: submit the CarDD license form and accept the Flickr/Shutterstock terms;
    accept the AGPL-3.0 obligations or obtain an Ultralytics Enterprise License.
  - Storage: CarDD on-disk size is **not reported** in the sources fetched; it is
    4,000 high-resolution images, so check the size when the download link is granted.
    YOLO11n/s checkpoints are small; exact file sizes not fetched.
  - Compute and time: not benchmarked. Training on the local Apple M4 via MPS is possible
    in principle; real epoch time must be measured on a short run before committing to a
    full fine-tune. A cloud GPU would mean paid compute, which needs a separate decision.
  - Unknowns to resolve first:
    - CarDD per-class counts and the severity spreadsheet contents.
    - VehiDE class names and license.
    - Whether a permissively licensed (non-AGPL) detector suits a commercial path.

## Could not verify

- VehiDE class names and license; the publisher page and DOAJ returned 403.
- Humans in the Loop Kaggle dataset details: the page did not render.
- Roboflow Universe project metrics and licenses: direct fetches returned 403, and licenses
  were seen only in search listings.
- Contents of the CarDD license PDF (only the paper's terms-of-use text was read), and
  CarDD download size.
- Class lists inside any community `best.pt`; this would require downloading them, which
  was not done.

## Sources

1. CarDD project page: https://cardd-ustc.github.io/
2. CarDD paper (HTML v2): https://arxiv.org/html/2211.00945v2
3. CarDD abstract: https://arxiv.org/abs/2211.00945
4. VehiDE follow-up article (DOAJ; 403 on fetch, cited from search results): https://doaj.org/article/6e0eb600059440b390431a42dfe8d138
5. VehiDE research page (search result): https://cvis.hcmute.edu.vn/Research/Details/4
6. Car Parts and Car Damages (Kaggle): https://www.kaggle.com/datasets/humansintheloop/car-parts-and-car-damages
7. COCO Car Damage Detection mirror: https://gts.ai/dataset-download/coco-car-damage-detection-dataset/
8. Roboflow Universe examples: https://universe.roboflow.com/cv-do/car-damage-c3mtk , https://universe.roboflow.com/r-0vhh2/cardamaged/dataset/
9. Small Dents, Big Impact: https://arxiv.org/abs/2508.15431
10. Ultralytics repository and license section: https://github.com/ultralytics/ultralytics
11. YOLO11 docs: https://docs.ultralytics.com/models/yolo11/
12. https://huggingface.co/nezahatkorkmaz/car-damage-level-detection-yolov8
13. https://huggingface.co/AItoolstack/car_damage_detection
14. https://huggingface.co/AItoolstack/car_damage_detection/blob/main/inference/damage_inference.py
15. https://huggingface.co/Mariofm02/YOLOv8m_Car_Parts_Damaged
16. https://huggingface.co/spaces/umerforsure/Car_Damage_Detection/blob/main/app.py
17. Hugging Face model search "car damage": https://huggingface.co/models?search=car%20damage
18. Ultralytics issue on model licensing: https://github.com/ultralytics/ultralytics/issues/2129
