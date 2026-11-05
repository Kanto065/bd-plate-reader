# Training pipeline

Trains the two small models behind BD Plate Reader and exports them as int8 TFLite for the phone.

1. Stage 1, plate detector: YOLO11n, one class (`license_plate`), input 640.
2. Stage 2, character detector: YOLO11n, 102 classes (digits, letters, Metro, district words), input 416.

The text is rebuilt from the detected characters by `postprocess.py`, a small pure Python module that Milestone 3 will port to TypeScript.

## Quick start on Colab (recommended)

You need a Google account. The free T4 GPU is enough.

1. Put the dataset zips in Google Drive at `My Drive/bd-plate-reader/data/`. The original names work, the scripts find the zips by name pattern:
   - `Bangla LPDB - A.zip` (required)
   - `NightLPD-BD Annotated Bangladeshi Nighttime Vehicl.zip` (required)
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

## Drive layout

```
My Drive/bd-plate-reader/
  data/       the zips above
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
- `BD_FRACTION`: use only a fraction of the training set (smoke tests).
- `BD_DEVICE`: for example `0` or `cpu`.
- `BD_PIPELINE_DIR`: folder of vehicle photos for the optional full pipeline test in `evaluate.py`. An optional `truth.csv` with columns `file,text` gives an exact match score, where text is the class names in reading order separated by spaces, for example `Dhaka Metro Ga 2 7 8 4 7 6`.

Defaults for each stage are in `config.py`.

## Scripts

- `organize_data.py`: finds the zips, extracts them (including the zip inside the NightLPD zip) into the work folder.
- `merge_datasets.py`: validates labels, builds `plate_detector/` and `char_detector/` (80, 10, 10 split with a fixed seed), writes `data.yaml` for each and `classes.txt`.
- `train.py plate|chars`: Ultralytics YOLO11n training with resume.
- `export.py`: int8 TFLite export, copies the files to `models/` and reports sizes against the 15 MB target.
- `evaluate.py`: mAP on the test split, plate level exact match, optional full pipeline test.
- `postprocess.py`: line grouping, left to right sorting, class mapping, plate string. Tests in `tests/`.
- `class_display_bn.json`: class name to Bangla text, used for the final plate string.
- `colab_train.ipynb`: all of the above in one notebook.

## Design choices

- No horizontal flip in augmentation. Bangla glyphs are not mirror symmetric.
- Stage 2 uses input 416 because it sees tight plate crops with up to about 11 characters. The smallest character stays around 30 pixels wide, enough to separate similar letters, while the model stays cheap on a phone.
- Stage 2 drops a whole plate if any one of its boxes is invalid, because a plate with a missing character label would teach a wrong sequence.
- Stage 1 merges daytime and night images, and each source is split separately so night images appear in train, val and test.

## What the evaluation does and does not tell you

- mAP is measured on a held out test split of the same datasets, so it is optimistic for photos taken in other conditions.
- The exact match score is at plate crop level (character model on ground truth crops). The vehicle photos in the dataset have no plate text, so a true end to end score needs your own labeled photos (`BD_PIPELINE_DIR` with `truth.csv`). Add 200 to 300 phone photos from Dhaka for this, as planned in CLAUDE.md.
- Several district classes have fewer than 20 training examples (see `docs/dataset_notes.md`). Expect weaker results on those.

## Licenses

Check each dataset license before any commercial use. See CLAUDE.md for the list.
