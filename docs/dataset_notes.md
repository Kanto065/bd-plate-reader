# Dataset notes

What the merge script found in the real data (checked on the downloaded zips, Milestone 2).

## Bangla LPDB-A

- Zip has one top folder, `Bangla LPDB - A`, with two folders inside.
- `Bangladeshi_Vehicle_Image_with_visible_license_plate`: 1,929 images with labels (mixed `.jpg`, `.jpeg`, `.png`), one class `LP`, one to three plates per image. No problems found.
- `Bangla_License_Plate`: 2,668 cropped plates with labels, `classes.txt` lists 102 names. Two to eleven characters per plate.

### Label problem in the character data

The labels use class ids up to 105, but `classes.txt` has only 102 names (ids 0 to 101). Three boxes use ids with no name:
- `1844.txt` has id 102
- `2570.txt` has id 103
- `2584.txt` has id 105

Id 104 is never used. These look like annotation mistakes. The merge script drops those three plates completely, leaving 2,665 plates for stage 2.

### Class names

The names in `classes.txt` are English transliterations, for example `Ka`, `Metro`, `Dhaka`, not Bangla script. The script keeps them exactly as given in `training/classes.txt`. The Bangla text comes from `training/class_display_bn.json`.

The letter names are irregular. They were checked against real crops:
- `Da` is ড, `DA` is দ, `Dha` is ঢ
- `Ta` is ট, `Tha` is থ, `THA` is ঠ
- `Ba` is ব, `Bha` is ভ, `A` is অ, `E` is এ, `U` is উ
- `Chatto` is the word চট্ট, `Raj` is the word রাজ (city words used with Metro)

The district spellings in the display map were written from general knowledge and not checked image by image. Verify them before relying on the Bangla output.

### Rare classes

Many districts are rare. After the 80, 10, 10 split, 31 classes have fewer than 20 training boxes. The rarest are Meherpur (8), Sa (10), Joypurhat (10), Kustia (10), Bandarban (11), Chandpur (11). The script makes sure every class appears at least once in train. More photos of these plates would help most.

## NightLPD-BD

- The outer zip holds one zip, `License_Plate_Detection.zip`. After unzipping, images are in `images/train`, `images/val`, `images/test` (401 in total).
- Labels come in four formats under `labels/`: `coco`, `pascal voc`, `yolo26-detection`, `yolo26-segmentation`. Only `yolo26-detection` is plain YOLO boxes, so that folder is used.
- 10 images have empty label files (no plate marked) and are dropped. 391 are used.
- The original train, val, test split is ignored. The images are merged with the daytime images and split again.

## Plate crops 2.5k (Bangla Licence plate detection and recognition dataset)

- Only cropped plate images, 5,254 files, no labels. Folders: `training`, `testing`, `new data` (JPG) and `training_data`, `testing_data` (black and white PNG versions).
- Not used for training. `evaluate.py` runs the character model on 50 crops from `testing` and writes `crops_check.csv` so you can judge it by eye.

## Resulting datasets

- Stage 1, plate detector: 2,320 images (train 1,856, val 232, test 232), one class.
- Stage 2, character detector: 2,665 plates (train 2,133, val 266, test 266), 102 classes, 22,168 boxes.
