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

Many districts are rare. After the 80, 10, 10 split, 28 classes have fewer than 20 LPDB-A training boxes. The rarest are Chandpur (8), Joypurhat (8), Sa (9), Meherpur (9), Kustia (10), Barguna (11), Netrokona (11). The script makes sure every class appears at least once in train. More photos of these plates would help most. The exact numbers depend on the split seed and are printed by `merge_datasets.py`.

## NightLPD-BD

- The outer zip holds one zip, `License_Plate_Detection.zip`. After unzipping, images are in `images/train`, `images/val`, `images/test` (401 in total).
- Labels come in four formats under `labels/`: `coco`, `pascal voc`, `yolo26-detection`, `yolo26-segmentation`. Only `yolo26-detection` is plain YOLO boxes, so that folder is used.
- 10 images have empty label files (no plate marked) and are dropped. 391 are used.
- The original train, val, test split is ignored. The images are merged with the daytime images and split again.

## DatasetNinja: Bangladeshi License Plate Recognition, Character Recognition

- Apache 2.0 license (source: a Kaggle dataset by Syed Nahin Hossain, exported through DatasetNinja).
- A tar file in Supervisely format: `train` (3,480 images), `test` (218), `validation` (363), each with `img/` and `ann/`, one JSON file per image named image name plus `.json`. All 27 classes are rectangles, character level, 4 to 9 boxes per image.
- Image names are Roboflow exports (`<source>.rf.<hash>.jpg`). Source groups (text before `.rf.`): 1,451 groups for 4,061 images. 870 groups have 4 copies (the original and three augmentations, all in the `train` folder) and 581 groups are single images (all of `test` and `validation`). No group spans more than one of the original folders. Many train copies are noisy or blurred augmentations.
- Conversion: boxes are clipped to the image (a few can cross the border) and an image is dropped if any class is unmapped or any box is degenerate. Nothing was dropped on the real data. The conversion was checked by drawing the converted boxes back on images.

### Class mapping to the unified 102 classes

Checked by eye on real plates from both datasets (`training/ninja_class_map.json`). Nothing was left unmapped.
- Digits `0` to `9` map to `0` to `9` (both are the Bangla digits ০ to ৯).
- `metro` to `Metro`, `dhaka` to `Dhaka`, `khulna` to `Khulna`, `jashore` to `Jessore` (the word যশোর).
- `chattogram` maps to **`Chatto`**, not `Chattogram`. The Ninja crops and plates show the short word চট্ট used on Chattogram Metro plates. Box shape agrees: median width to height ratio 1.14 in Ninja, 1.04 for LPDB-A `Chatto`, 1.95 for LPDB-A `Chattogram`.
- Letters: `ba` to `Ba` (ব), `ca` to `Cha` (চ), `da` to `Da` (ড), `ga` to `Ga` (গ), `gha` to `Gha` (ঘ), `ha` to `Ha` (হ), `ja` to `Ja` (জ), `ka` to `Ka` (ক), `kha` to `Kha` (খ), `na` to `Na` (ন), `ta` to `Ta` (ট), `va` to `Bha` (ভ).
- Remaining doubt: many Ninja crops are noisy, so the decision for each letter rests on the cleanest plates, three to six per letter. No letter was confusing on those. A Ninja letter that does not exist in LPDB-A (for example ত or ধ) would show up as a mislabeled class, and none was seen.

### Split by source photo

- Groups are shuffled with the fixed seed 42 and assigned whole. About 30 percent of the images are held out for testing, then about 10 percent of the remaining images go to val. The merge script stops with an error if any group appears on both sides of any split.
- Result: train 2,554 images from 928 groups, val 288 from 102 groups, test 1,219 from 421 groups (30.0 percent of images). 266 of the 421 test groups have four copies each.
- The held out Ninja test set (`images/test_ninja`) is evaluated separately from the LPDB-A test split.

### Effect on class coverage

Ninja has only 27 classes and they are mostly the common ones. It helps the letters Ja (LPDB-A train 21, Ninja 198), Kha (34, 123), Ba (47, 244), Ta (97, 260), Ga (299, 1,163), Gha (83, 152), Khulna (67, 257), Jessore (38, 102), and the digits, Metro and Dhaka. It adds nothing for any of the 28 rare classes (district words such as Chandpur, Meherpur and Kustia, and letters Sa and Pa). Its `chattogram` goes to `Chatto`, so the rare LPDB-A `Chattogram` is not helped.

Ninja images also make up about two thirds of the stage 2 training images, many of them noisy augmented copies.

## Plate crops 2.5k (Bangla Licence plate detection and recognition dataset)

- Only cropped plate images, 5,254 files, no labels. Folders: `training`, `testing`, `new data` (JPG) and `training_data`, `testing_data` (black and white PNG versions).
- Not used for training. `evaluate.py` runs the character model on 50 crops from `testing` and writes `crops_check.csv` so you can judge it by eye.

## Resulting datasets

- Stage 1, plate detector: 2,320 images (train 1,856, val 232, test 232), one class.
- Stage 2, character detector: LPDB-A 2,665 plates (train 2,133, val 266, test 266), 102 classes, 22,168 boxes, plus DatasetNinja. Combined: train 4,687 images, val 554, LPDB-A test 266, and a separate Ninja test of 1,219 images.
