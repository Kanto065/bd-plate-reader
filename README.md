# BD Plate Reader

An Android app that reads a Bangladeshi car number plate from a photo and shows the result as editable, copyable text. Take a picture or pick one from the gallery. Front or back plates, one or two lines, Bangla script (for example ঢাকা মেট্রো-ঘ ১১-২৩৪৫).

Everything runs on the phone with small TFLite models. It works offline and no photo is uploaded anywhere. Plates can be dirty, angled or shot at night, so always check the text before copying. See CLAUDE.md for the full plan and working rules.

## Install the app

Download the latest APK from the [Releases page](https://github.com/Kanto065/bd-plate-reader/releases). Copy it to an Android phone, allow installs from unknown sources, and open it.

- Android arm64 only.
- The APK is signed with the public debug key. Use it on your own phone for testing. It is not a Play Store build.

## How it works

1. A native Android module decodes the photo, applies the EXIF rotation and letterboxes it to 640 px.
2. Stage 1, plate detector: YOLO11n, int8 TFLite, one class. The best box is cropped from the full photo with a small margin and letterboxed to 416 px.
3. Stage 2, character detector: YOLO11n, int8 TFLite, 102 classes (Bangla letters, digits, district and metro words). No OCR engine.
4. Post-processing in TypeScript: group the characters into lines by y, sort each line by x, map class IDs to Bangla text, and flag low confidence.

The models are about 6 MB together and load when the app opens. On a Galaxy A12, build 16 reads a photo in about 2.3 s, down from 11 s for the first version. Per stage timings are logged as `PLATE_TIMING` in logcat.

## Results so far

Float models on held-out test sets:

| Model | mAP50 | mAP50-95 |
|---|---|---|
| Plate detector (232 images) | 0.962 | 0.637 |
| Character detector, LPDB-A test | 0.975 | 0.684 |
| Character detector, DatasetNinja test | 0.960 | 0.692 |

Exact whole-plate match on cropped plates is 84.6% (LPDB-A) and 79.2% (DatasetNinja). The int8 files and the full photo-to-text pipeline have no scored test set yet. Details are in `docs/PROJECT_STATE.md`.

## Repo layout

- `app/` React Native app (bare workflow, TypeScript). `app/src/ml/` holds the pipeline, `app/android/` the native image module.
- `training/` Python scripts, dataset configs and Colab notebooks. Datasets under `training/data/` are gitignored.
- `models/` exported .tflite files. The app bundles its own copies in `app/assets/models/`.
- `docs/` project state, dataset notes and ML integration notes.

## Build and test

Tests, from `app/`:
```
npm install
npm test
npx tsc --noEmit
npm run check:classes
```
Python tests, from `training/`: `python -m pytest`.

`npm install` also applies a patch to react-native-fast-tflite (`app/patches`). It fixes model loading in release builds, moves LiteRT to 1.4.2 and sets the thread count. The reasons are in `docs/ml_integration.md`.

### Run on a phone

One time setup:
1. Install JDK 17 (Temurin) and the Android SDK with NDK 27.1.12297006. Set `JAVA_HOME` and `ANDROID_HOME`, and add `platform-tools` to `PATH`.
2. On the phone, enable USB debugging, connect it, accept the prompt, and check that `adb devices` shows `device`.

Then from `app/`: `npm install`, `npx react-native start`, and in a second terminal `npx react-native run-android`.

Notes:
- The build is limited to arm64-v8a (`reactNativeArchitectures` in `app/android/gradle.properties`). Add `x86_64` to use an emulator.
- Create `app/android/local.properties` with `sdk.dir=<path to the SDK>` if Gradle cannot find it. It is not committed.
- On Windows the native build can fail with `Filename longer than 260 characters` when the project sits in a deep folder such as OneDrive. Clone to a short path like `C:\bdp`, or let CI build it.
- iOS strings are configured, but iOS is not built or tested.

## CI and releases

Work happens on a branch for each feature or fix. Merging into `main` runs GitHub Actions (`.github/workflows/android-build.yml`): tests and type check, then a debug APK and a standalone release APK. On `main` the release job also publishes the APK on the Releases page as a pre-release. Branches do not publish releases.

## Docs

- `docs/PROJECT_STATE.md` handoff notes, results and next steps.
- `docs/ml_integration.md` model shapes, quantization, the pipeline and the library patch.
- `docs/dataset_notes.md` datasets and their licenses.
- `training/README.md` training and export on Colab or a local GPU.

## Licenses

The training datasets have their own licenses. Check each one before any commercial use.
