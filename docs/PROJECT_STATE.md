# BD Plate Reader: project state and handoff

Updated 2026-11-08 so the project can be cloned and continued on another PC. Read CLAUDE.md first for the rules (no em dashes or double hyphens in text, on-device only, small tested functions, branch per feature or fix, CI and releases from main only).

## What it is

A React Native (bare, TypeScript) Android app. The user takes a photo or picks one from the gallery. The app finds a Bangladeshi BRTA number plate (Bangla script, one or two lines), reads it, and shows editable, copyable text. Everything runs on the phone, offline. Latest release: build 23 on the GitHub Releases page.

Pipeline:
1. A native Android module (`PlateImageModule.kt`) decodes the photo with BitmapFactory, applies the EXIF rotation, letterboxes it to 640 px and writes the model input as planar float32 to a cache file. JavaScript reads the file as a raw buffer.
2. Stage 1: YOLO11n plate detector, 1 class. The best box is cropped from the full photo (up to 2048 px) with 5 percent margin and letterboxed to 416 px.
3. Stage 2: YOLO11n character detector, 102 classes (digits, series letters, district and Metro words).
4. Post-processing (`app/src/ml/postprocess.ts`, Python twin `training/postprocess.py`): estimate the plate tilt from the character centers and level it, group the characters into lines by y, sort by x, map class IDs to Bangla text. `confidence.ts` decides when to show "Check the characters before copying" and lists the reasons.

The models in `app/assets/models` are **dynamic range** exports (int8 weights, float activations, about 3 MB each) that keep the `_int8` file names. See `docs/ml_integration.md`.

## Results

### Held out test sets from training (float models, 2026-11-05)

| Model | mAP50 | mAP50-95 |
|---|---|---|
| Plate detector (test, 232 images) | 0.962 | 0.637 |
| Character detector, LPDB-A test | 0.975 | 0.684 |
| Character detector, DatasetNinja test | 0.960 | 0.692 |

Whole-plate exact match on cropped plates: 84.6% (LPDB-A test), 79.2% (Ninja test), mean character error rate about 4%.

### Full photo to text on photos the models never saw (2026-11-08)

64 hand labeled photos (bikes and private cars) from the Mendeley dataset "Vehicle License Plate Detection Dataset V1" (CC BY 4.0, DOI 10.17632/nntnnzffw4). Only plates a person could read were labeled, so the numbers are slightly optimistic. Labels are in `training/eval_data/mendeley_v1_truth64.csv`, the photos are rebuilt with `training/make_mendeley_sample.py`.

| Models | Reader | Exact match | Character error rate |
|---|---|---|---|
| Full int8 | old (no tilt fix) | 78.1% | 8.4% |
| Full int8 | tilt aware | 87.5% | 3.3% |
| Dynamic range | old (no tilt fix) | 85.9% | 5.8% |
| Dynamic range | tilt aware | 95.3% | 1.6% |
| fp32 reference | tilt aware | 95.3% | 1.6% |
| **Dynamic range, tilt aware, on a Galaxy A12 (build 23)** | | **96.9%** | n/a |

The tilt aware reader and the dynamic range models each add about 8 to 9 points. All rows except the last ran on a PC with the same photos.

Remaining mistakes: Ga read as Kha or Gha (2 plates, confident, no score can flag them) and one steep tilt.

### Why dynamic range

The full int8 export crushed the confidence output. The plate score was exactly 0.50 on every photo with about 9 candidates tied, and a character crop had 8 to 18 distinct scores. NMS could not rank tied candidates, so a photo with a second, smaller plate behind the car was read from an arbitrary plate. Dynamic range gives 0.77 to 0.85 plate scores with one winner and about 99 distinct character scores, and matches fp32 exactly. The new Ultralytics LiteRT exporter has no fp16.

### Tilt

Training rotation was only 3 degrees for the character detector and 5 for the plate detector. The detector still finds the characters on a tilted plate, but grouping by y assumed a level plate. `estimateTilt` now tries angles from -35 to 35 degrees and keeps the smallest one that gives one or two tight rows. On 20 photos rotated in software, correct reads at 15 degrees went from 3 to 15. A steep tilt (around 20 degrees) on a small plate can still fail because the detector misses characters.

### Speed on a Galaxy A12

| Build | What changed | Per photo |
|---|---|---|
| First | JavaScript JPEG decode, resize and rotate | 11.1 s |
| 15 | Native decode, models preloaded | 3.8 s |
| 16 | Native letterbox and crop, model input as a file | 2.3 s |
| 17 | 4 CPU threads, decoded photo reused | 1.2 s |
| 22 and later | Dynamic range models | 1.4 s (median, max 2.25 s) |

## Branches and releases

Work on a branch per feature or fix, merge into `main`, and CI (`.github/workflows/android-build.yml`) builds on `main`: tests and type check, a debug APK, a standalone release APK, and a GitHub pre-release. Promote a build to Latest by hand with `gh release edit build-N --prerelease=false --latest`. The APK is signed with the committed debug keystore, for testing on your own phone only.

All feature branches are merged into `main`. The only unmerged branch is `finish-notebook`, which holds `training/colab_finish.ipynb` (export and evaluate without training). Its export fix is already in `main`.

## Tools

| Tool | Use |
|---|---|
| `training/pipeline_eval.py` | Run the shipped models on a photo folder on the PC, writes `predictions.csv`, scores against `truth.csv`. Needs ai-edge-litert, pillow, numpy. `BD_MODEL_SUFFIX` picks int8, dynamic or fp32. |
| `training/phone_selftest.py` | Same photos through the installed app on a USB phone, no photo picker. Uses the test only link `bdplate://selftest`. |
| `training/export.py` and `training/colab_export_modes.ipynb` | Export `int8`, `dynamic` or `fp32` models from the saved weights on Colab. |
| `training/make_mendeley_sample.py` | Rebuild the labeled test photos from the public dataset. |

## Not verified or weak

- Only white plates of bikes and private cars were labeled. Green and teal plates (CNGs, buses, many trucks) read much worse in a first look (only 20 to 42 percent had a valid plate format) and are unmeasured.
- The test photos are 640 px, so plates are small. Real phone photos have more pixels, so real use should be easier.
- Ga, Kha and Gha are confused. More labeled examples of these series letters are needed.
- 83 of 147 plates were too blurry to label and are not in the score.
- 28 character classes have under 20 training boxes (for example Chandpur, Joypurhat, Meherpur, Kustia). They will be weak.
- Bangla spellings in `training/class_display_bn.json` for district names are unchecked.
- The "Check the characters" banner appeared on 4 of 64 photos and all 4 were correct. It catches no wrong plates on this set.
- GPU or NNAPI delegates were not tried.

## Next steps

1. Collect 200 to 300 own Dhaka photos with the true plate text. Run `pipeline_eval.py` to pre-fill `predictions.csv`, correct only the wrong rows, and score. Include green plates, tilt, distance and night.
2. Retrain on Colab with stronger rotation and perspective augmentation (degrees 10 to 15), more examples of the confused series letters and the rare classes, and the new photos. Export with `BD_EXPORT_DTYPE=dynamic`.
3. Label and score the green and teal plates, then decide whether they need their own training data.
4. Re-check the confidence rule on the larger set. Keep live-frame detection out of scope.

## Training notes

- Ultralytics 8.4.x, YOLO11n, no horizontal flip, mosaic 0.3, close_mosaic 20. Stage 1: 100 epochs, batch 16, patience 30. Stage 2: 150 epochs, batch 32, patience 40 (stopped early at epoch 113, best epoch 73).
- Data on Colab from a Drive folder `bd-plate-reader/data`: Bangla LPDB-A zip, NightLPD-BD zip, the 2.5k crops zip, and the DatasetNinja tar (Supervisely JSON, split by source group so copies never straddle train and test). NightLPD-BD and LPDB-A were merged and re-split for stage 1, so none of their images are an unseen test.
- Ultralytics 8.4.x redirects `format="tflite"` to the LiteRT exporter, which writes next to `best.pt`. `training/export.py` handles both layouts.
- Int8 calibration used only 232 plate val images.
- Colab flow: `training/colab_train.ipynb` to train, `training/colab_export_modes.ipynb` to export. Outputs go to Drive `bd-plate-reader/outputs/` (`models/`, `eval_results.json`).

## Set up on a new PC

Android app (Windows, macOS or Linux):
1. Install JDK 17 (Temurin). JDK 25 breaks React Native builds.
2. Install Node 20 or newer, Android Studio or the SDK command line tools, and set `ANDROID_HOME`. Project NDK is 27.1.12297006.
3. `cd app && npm install` (this also applies `app/patches/react-native-fast-tflite+3.0.1.patch`).
4. Tests: `npm test`, `npx tsc --noEmit`, `npm run check:classes`.
5. Phone: enable USB debugging, `adb devices`, then install a CI APK from the Releases page. Local builds on Windows can fail with `Filename longer than 260 characters` in a deep folder such as OneDrive, so let CI build.

Training side:
1. Python 3.10 or newer, `pip install -r training/requirements.txt`. Export needs `training/requirements-export.txt`. Evaluation scripts need only `ai-edge-litert pillow numpy`.
2. Tests: `cd training && python -m pytest`.
3. Use Colab for training and export (free T4). Datasets are gitignored under `training/data/`.

## Where things are

- Models in the app: `app/assets/models/*.tflite`. Copies of all exports on Drive `bd-plate-reader/outputs/models/` (int8, dynamic, fp32).
- Class list: `training/classes.txt` (102). TypeScript copy `app/src/ml/classes.ts`, generated by `app/scripts/gen-classes.js`.
- Post-processing: `app/src/ml/postprocess.ts`, `confidence.ts`, `yolo.ts`. Python twin: `training/postprocess.py`. Keep them identical.
- Native image code: `app/android/app/src/main/java/com/bdplatereader/PlateImageModule.kt`.

## Known pitfalls

- react-native-worklets-core 1.6.3 crashes on the Galaxy A12 (`__cxa_init_primary_exception`, probably NDK mismatch). It is removed.
- vision-camera is pinned to v4. Keep the camera active during capture (see `CameraScreen.tsx`).
- react-native-fast-tflite needed a patch for release builds, LiteRT 1.4.2 and a thread count. See `docs/ml_integration.md`.
- Git Bash rewrites `/sdcard/...` paths for adb. Set `MSYS_NO_PATHCONV=1`, or use `phone_selftest.py`.
- Python on Windows cannot read Git Bash `/tmp` paths. Use `cygpath -w`. `python -I` ignores `PYTHONPATH`.
- Do not drive the phone's photo picker with taps. Its gallery shows personal photos. Use the self test.
- Large downloads and heavy installs: ask first (see CLAUDE.md).
