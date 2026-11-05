# Project: BD Plate Reader

## Goal
A React Native mobile app that reads a Bangladeshi car number plate from a photo and shows the result as plain, copyable text. The user either takes a picture with the camera or uploads one from the gallery. The plate can be on the front or the back of the car. Inference runs fully on-device with small TFLite models, so the app works offline and no image is uploaded anywhere.

## Target plates
- BRTA plates in Bangla script, for example: ঢাকা মেট্রো-ঘ ১১-২৩৪৫ (district or metro name, series letter, then digits).
- Both single-line and two-line layouts must work.
- Some plates are not clean (dirt, angle, glare, night). Always let the user edit the text before copying.

## Stack
- React Native (bare workflow or Expo dev build, NOT Expo Go), TypeScript.
- react-native-vision-camera pinned to v4 (the TFLite plugins do not target v5 yet).
- react-native-fast-tflite for inference, vision-camera-resize-plugin for resizing.
- react-native-image-picker for gallery upload.
- @react-native-clipboard/clipboard for the copy button.
- Training side: Python, Ultralytics YOLO (YOLO11n), exported to int8 TFLite. Train on Colab or a local GPU.

## Model pipeline
1. Stage 1, plate detector: YOLO11n, 1 class (license plate). Crop the best box with a small padding.
2. Stage 2, character detector: YOLO11n on the cropped plate, about 102 classes (Bangla letters, digits, district and metro words). OCR-free design.
3. Post-processing: group detected characters into lines by y-coordinate, sort each line left to right by x-coordinate, map class IDs to text, join the lines. Optionally also output a transliterated English form.
4. Keep both models quantized (int8) and small. Target total model size under 15 MB.

## Datasets (check each license before any commercial use)
- Bangla LPDB-A (Zenodo record 4718238): 1,928 vehicle images, 2,662 cropped plates, YOLO format, 102 character classes in part 2.
- Bangla License Plate Dataset 2.5k (Zenodo record 7110401): 2,519 images with cropped plates.
- Bangladeshi License Plate Recognition, Character Recognition (Dataset Ninja): 4,061 images, Apache 2.0.
- NightLPD-BD (Mendeley Data, DOI 10.17632/bnw8r23dgf.1): 401 night images, YOLO format.
- Roboflow Universe projects: search "Bangla license plate" and "Bangladeshi license plate", merge the good ones.
- Later: add 200 to 300 own phone photos taken in Dhaka for real-world robustness.

## Repo layout
- `app/` React Native app
- `training/` Python scripts, dataset configs, notebooks
- `training/data/` datasets (gitignored)
- `models/` exported .tflite files copied into the app assets
- `docs/` notes and evaluation results

## Working rules
- Ask before downloading large datasets or installing heavy tooling, and state size and source.
- Never upload user photos to any server. On-device only.
- Keep the app simple: Home (Camera / Upload), Result (editable text, Copy button, Retake).
- Write small, testable functions for the post-processing logic (line grouping, sorting, class mapping) and add unit tests for them.
- Do not use em dashes or double hyphens in any text, comments, docs or UI copy. Use commas or periods.
- After each milestone, summarize what works and what is still weak.
