# BD Plate Reader: project state and handoff

Written 2026-11-05 so the project can be cloned and continued on another PC. Read CLAUDE.md first for the rules (no em dashes or double hyphens in text, on-device only, small tested functions).

## What it is

A React Native (bare, TypeScript) Android app. The user takes a photo or picks one from the gallery. The app finds a Bangladeshi BRTA number plate (Bangla script, one or two lines), reads it, and shows editable, copyable text. Everything runs on the phone, offline.

Pipeline:
1. Stage 1: YOLO11n plate detector, int8 TFLite, input 640 px, 1 class.
2. Crop the best box with small padding.
3. Stage 2: YOLO11n character detector, int8 TFLite, input 416 px, 102 character classes.
4. Post-processing: group characters into lines by y, sort by x, map class IDs to Bangla text, flag low confidence.

## Branches

- `main`: Milestones 1 to 3 (UI, training pipeline, integration skeleton). Placeholder text when no models are present.
- `integrate-models`: real models linked into the app (`app/assets/models/`, force-added), NCHW input fix, tests. CI passes. Not merged yet. Merge it after testing on a phone.
- `finish-notebook`: `training/colab_finish.ipynb` (export and evaluate without training) and the `export.py` fix for the LiteRT exporter.
- `release-step`: CI step that publishes the release APK as a GitHub pre-release (works, see Releases page).

## Results so far

Held-out test sets, measured on the float `.pt` models:

| Model | mAP50 | mAP50-95 |
|---|---|---|
| Plate detector (test, 232 images) | 0.962 | 0.637 |
| Character detector, LPDB-A test | 0.975 | 0.684 |
| Character detector, DatasetNinja test | 0.960 | 0.692 |

Whole-plate exact match on cropped plates: 84.6% (LPDB-A test), 79.2% (Ninja test). Mean character error rate about 4%. Common misses: a duplicated or dropped digit, swapped neighbouring digits, wrong district word.

Exported models: plate 3.06 MB, char 3.08 MB, total 6.14 MB (target under 15 MB). The int8 files were not separately scored.

Real tensors (inspected with ai-edge-litert): inputs are planar NCHW float32 `[1,3,640,640]` and `[1,3,416,416]`. Outputs `[1,5,8400]` and `[1,106,3549]`, float32, boxes normalized 0..1 as center x, y, w, h, scores already sigmoided. See `docs/ml_integration.md`.

## Not verified yet

- The APK has not been installed with the real models on a phone. Unknown: whether react-native-fast-tflite loads the LiteRT-exported files, speed, EXIF/rotation handling from the picker.
- The full photo-to-text pipeline has no scored test set. It needs photos with known plate text, ideally 200 to 300 own Dhaka photos.
- Confidence thresholds (plate score 0.3, char score 0.6, 5 to 6 digits) need tuning on real photos.
- 28 character classes have under 20 training boxes (for example Chandpur, Joypurhat, Meherpur, Kustia). They will be weak.
- Bangla spellings in `training/class_display_bn.json` for district names are unchecked.
- Plate crops come from a JPEG decoded to at most about 1600 px, so a small distant plate may be unreadable.

## Training notes

- Ultralytics 8.4.202, YOLO11n, no horizontal flip, mosaic 0.3, close_mosaic 20. Stage 1: 100 epochs, batch 16, patience 30. Stage 2: 150 epochs, batch 32, patience 40 (stopped early at epoch 113, best epoch 73).
- Data on Colab from a Drive folder `bd-plate-reader/data`: Bangla LPDB-A zip, NightLPD-BD zip, the 2.5k crops zip, and the DatasetNinja tar (Supervisely JSON, split by source group so copies never straddle train and test).
- Ultralytics 8.4.x redirects `format="tflite"` to the LiteRT exporter. It writes `best_int8.tflite` next to `best.pt`, not into `best_saved_model/`. `training/export.py` handles both.
- Int8 calibration used only 232 plate val images (300 or more recommended).
- Colab flow: open `training/colab_train.ipynb` to train, or `training/colab_finish.ipynb` to export and evaluate from saved `best.pt` files. Outputs go to Drive `bd-plate-reader/outputs/` (`models/`, `eval_results.json`).

## Set up on a new PC

Android app (Windows, macOS or Linux):
1. Install JDK 17 (Temurin). JDK 25 breaks React Native builds.
2. Install Node 20 or newer, Android Studio or the SDK command line tools, and set `ANDROID_HOME`. Project NDK is 27.1.12297006.
3. `cd app && npm install`
4. Tests: `npm test`, `npx tsc --noEmit`, `npm run check:classes`.
5. Phone: enable USB debugging, `adb devices`, then either install a CI APK or run `npx react-native run-android` with Metro (`adb reverse tcp:8081 tcp:8081`). Debug builds need Metro, release builds are standalone.

Preferred way to get an APK: push a branch and let GitHub Actions build it (`.github/workflows/android-build.yml`, jobs test, debug, release). On `main` and `release-step`, the release job also publishes the APK on the repo's Releases page as a pre-release. The APK is signed with the committed debug keystore, for testing on your own phone only.

Training side:
1. Python 3.10 or newer, `python -m venv .venv`, `pip install -r training/requirements.txt` (export needs `training/requirements-export.txt`).
2. Tests: `cd training && python -m pytest`.
3. Use Colab for training and export (free T4). Datasets are gitignored under `training/data/`.

## Where things are

- Models in the app: `app/assets/models/*.tflite` (on `integrate-models`). Copies in `models/` locally and on Drive `bd-plate-reader/outputs/models/`.
- Class list: `training/classes.txt` (102). TypeScript copy `app/src/ml/classes.ts`, generated by `app/scripts/gen-classes.js`.
- Post-processing: `app/src/ml/postprocess.ts`, `confidence.ts`, `yolo.ts`. Python twin: `training/postprocess.py`.

## Next steps

1. Install the `integrate-models` APK on a phone and test with real photos (gallery and camera). Report crashes and wrong reads.
2. If the models do not load, check the logcat for TFLite op or version errors.
3. Measure time per stage on a low-end phone (Galaxy A12, arm64).
4. Tune thresholds, then merge `integrate-models` into `main`.
5. Collect 200 to 300 own Dhaka photos with truth text, build a full-pipeline test set, retrain with them and with more data for rare classes, use 300 or more calibration images.
6. Speed work if needed: native image decode and resize, GPU or NNAPI delegate versus XNNPACK. Measure first. Skip live-frame detection for now.

## Known pitfalls

- react-native-worklets-core 1.6.3 crashes on the Galaxy A12 (`__cxa_init_primary_exception`, probably NDK mismatch). It is removed. If you re-add worklets or the resize plugin, try NDK 28.
- vision-camera is pinned to v4. Keep the camera active during capture (see `CameraScreen.tsx`).
- Python on Windows cannot read Git Bash `/tmp` paths. Use `cygpath -w`.
- Large downloads and heavy installs: ask first (see CLAUDE.md).
