# Training pipeline

Trains the two small models behind BD Plate Reader and exports them as int8 TFLite for the phone.

1. Stage 1, plate detector: YOLO11n, one class (`license_plate`), input 640.
2. Stage 2, character detector: YOLO11n, 102 classes (digits, letters, Metro, district words), input 416.

The text is rebuilt from the detected characters by `postprocess.py`, a small pure Python module that Milestone 3 will port to TypeScript.

## Quick start on Colab (recommended)

You need a Google account. The free T4 GPU is enough.

1. Put the dataset archives in Google Drive at `My Drive/bd-plate-reader/data/`. The original names work, the scripts find the files by name pattern:
   - `Bangla LPDB - A.zip` (required)
   - `NightLPD-BD Annotated Bangladeshi Nighttime Vehicl.zip` (required)
   - `bangladeshi-license-plate-recognition_-character-recognition-DatasetNinja.tar` (recommended, extra character data and an independent test set)
   - `Bangla Licence plate detection and recognition dataset.zip` (optional, no labels, only used for a visual check)
2. Open `training/colab_train.ipynb` in Colab. Upload it, or open it from GitHub with File, Open notebook, GitHub tab.
3. Runtime, Change runtime type, GPU.
4. Run all cells from top to bottom. Colab asks for permission to access Drive.
5. Results land in `My Drive/bd-plate-reader/outputs/`:
   - `plate_detector/` and `char_detector/` with `weights/best.pt`, `weights/last.pt` and training plots
   - `models/plate_detector_int8.tflite` and `models/char_detector_int8.tflite`
   - `eval_results.json`, plus `crops_check.csv` (predictions on unlabeled crops for checking by eye)

If Colab disconnects, open the notebook again and run the cells from the top. Dataset extraction is repeated (a few minutes), and training resumes from the last saved epoch. A finished stage is skipped.

If the GitHub repo is private, the clone cell fails. Either make it public for the run, or upload the `training/` folder to Drive and change the clone cell to copy it.

## Finishing early: export and evaluate without training

If you stop stage 2 before it ends (for example you have to leave), the `best.pt` of each stage is already saved on Drive. Use `training/colab_finish.ipynb` to export and evaluate from those files. It never trains.

1. Open `training/colab_finish.ipynb` in Colab (GitHub link: `https://colab.research.google.com/github/Kanto065/bd-plate-reader/blob/finish-notebook/training/colab_finish.ipynb`, use `main` in place of `finish-notebook` once the branch is merged).
2. Runtime, Change runtime type, GPU. Run all cells from top to bottom.
3. It rebuilds the datasets (needed for the int8 calibration images and the test sets), prints the size and save time of both `best.pt` files and the last epoch logged, then runs `export.py` and `evaluate.py`.
4. Download `models/plate_detector_int8.tflite`, `models/char_detector_int8.tflite` and `eval_results.json` from `My Drive/bd-plate-reader/outputs/`.

`export.py` and `evaluate.py` read `best.pt` directly from the outputs folder. They do not need `last.pt` or a finished marker. A character detector from an interrupted run is a valid but partly trained model, so the scores will be lower than after a full run. To resume training later, run `colab_train.ipynb` again. It continues from `last.pt`.

## Drive layout

```
My Drive/bd-plate-reader/
  data/       the archives above (zip and tar)
  outputs/    created by the scripts
```

## Running locally

Training is slow without an NVIDIA GPU, so Colab is the better place. The data scripts and tests run anywhere.

```
cd training
pip install -r requirements.txt
python organize_data.py "C:\path\to\folder\with\zips"
python merge_datasets.py
python -m pytest -q
python train.py plate
python train.py chars
python export.py
python evaluate.py
```

Run the TFLite export on Linux or Colab. It needs TensorFlow related packages (`requirements-export.txt`) and is fragile on Windows.

## Settings

Environment variables, all optional:
- `BD_DATA_DIR`: where the zips are. Default on Colab is the Drive folder above, locally `training/data/zips`.
- `BD_WORK_DIR`: scratch space for extracted and merged data. Default `/content/work` on Colab, locally `training/data/work` (git ignored).
- `BD_OUT_DIR`: runs, weights and exports. Default is the Drive `outputs` folder on Colab, locally `training/runs`.
- `BD_EPOCHS`, `BD_BATCH`: override epochs and batch size.
- `BD_NINJA_TEST_ONE_PER_GROUP`: set to `1` to keep only one image per source photo in the held out Ninja test set (default keeps all copies, so about 30 percent of the Ninja images are held out).
- `BD_FRACTION`: use only a fraction of the training set (smoke tests).
- `BD_DEVICE`: for example `0` or `cpu`.
- `BD_PIPELINE_DIR`: folder of vehicle photos for the optional full pipeline test in `evaluate.py`. An optional `truth.csv` with columns `file,text` gives an exact match score, where text is the class names in reading order separated by spaces, for example `Dhaka Metro Ga 2 7 8 4 7 6`.

Defaults for each stage are in `config.py`.

## Scripts

- `organize_data.py`: finds the zips and the tar, extracts them safely (including the zip inside the NightLPD zip) into the work folder.
- `merge_datasets.py`: validates labels, builds `plate_detector/` and `char_detector/` (LPDB-A split 80, 10, 10 with a fixed seed), writes `data.yaml` for each and `classes.txt`. For the character set it adds the DatasetNinja photos and writes `data_test_ninja.yaml` for the held out Ninja test set.
- `ninja_convert.py`: reads the Supervisely format of DatasetNinja, converts boxes to YOLO, groups images by source photo and splits by group. `ninja_class_map.json` maps the 27 Ninja class names to the unified class names, checked by eye (see `docs/dataset_notes.md`).
- `train.py plate|chars`: Ultralytics YOLO11n training with resume.
- `export.py`: int8 TFLite export, copies the files to `models/` and reports sizes against the 15 MB target.
- `evaluate.py`: mAP and plate level exact match on both held out character sets (LPDB-A test split and DatasetNinja held out photos, reported separately), plus an optional full pipeline test.
- `postprocess.py`: line grouping, left to right sorting, class mapping, plate string. Tests in `tests/`.
- `class_display_bn.json`: class name to Bangla text, used for the final plate string.
- `colab_train.ipynb`: all of the above in one notebook.

## Design choices

- No horizontal flip in augmentation. Bangla glyphs are not mirror symmetric.
- Stage 2 uses input 416 because it sees tight plate crops with up to about 11 characters. The smallest character stays around 30 pixels wide, enough to separate similar letters, while the model stays cheap on a phone.
- Stage 2 drops a whole plate if any one of its boxes is invalid, because a plate with a missing character label would teach a wrong sequence.
- Stage 1 merges daytime and night images, and each source is split separately so night images appear in train, val and test.
- DatasetNinja images come from Roboflow, where one source photo appears as up to four copies (the original and augmentations). The split is done per source photo (the file name text before `.rf.`), never per image, so no copy of a held out photo is used for training. About 70 percent of the Ninja images go to stage 2 training (a small part of them to val) and about 30 percent are held out as a second, independent test set.
- Ninja contributes about two thirds of the stage 2 training images, and many of them are heavily augmented copies. If the character model looks over fitted to the Ninja look, try training on LPDB-A plus only one copy per Ninja source photo.

## What the evaluation does and does not tell you

- mAP is measured on held out test sets, so it is optimistic for photos taken in other conditions. The LPDB-A test split comes from the same source as the training plates. The DatasetNinja test set is a different distribution, so its numbers are the more honest ones. Expect them to be lower.
- The exact match score is at plate crop level (character model on ground truth crops). The vehicle photos in the dataset have no plate text, so a true end to end score needs your own labeled photos (`BD_PIPELINE_DIR` with `truth.csv`). Add 200 to 300 phone photos from Dhaka for this, as planned in CLAUDE.md.
- 28 district and letter classes have fewer than 20 LPDB-A training examples (see `docs/dataset_notes.md`). DatasetNinja does not help with these, because its 27 classes are mostly common ones. Expect weaker results on the rare districts.

## Licenses

Check each dataset license before any commercial use. See CLAUDE.md for the list.
