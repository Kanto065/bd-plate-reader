# On-device ML integration

This note describes how the app runs the two TFLite models, what it assumes
about them, and what to check once the real exported models arrive. Everything
described as "assumed" has not been verified against a real model yet, because
the models were still training when this was written.

## Pipeline

```
photo (JPEG) -> decode -> upright RGB
  -> letterbox to 640 x 640 -> plate detector -> NMS -> best plate box
  -> pad the box by 5 percent, crop
  -> letterbox to 416 x 416 -> character detector -> NMS
  -> post-processing (lines by y, sort by x, map to Bangla) -> text
  -> confidence rules -> low confidence flag
```

Code map, all under `app/src`:

| File | Job |
|---|---|
| `services/readPlate.ts` | Entry point used by the Result screen. Falls back to the placeholder when no models are bundled. |
| `ml/plateReader.ts` | The pipeline. No native imports, tested with fake models. |
| `ml/modelLoader.ts` | Loads the models with react-native-fast-tflite. Required lazily. |
| `ml/modelAssets.ts` | Generated. `null` per model until the files are linked. |
| `ml/imageLoader.ts` | Reads the file, decodes the JPEG with jpeg-js, applies EXIF orientation. |
| `ml/imageOps.ts` | Letterbox, crop, EXIF, JPEG header parsing. |
| `ml/yolo.ts` | `decodeYolo` and `nms`. |
| `ml/quant.ts` | int8 and uint8 quantize and dequantize. |
| `ml/postprocess.ts` | Port of `training/postprocess.py`. |
| `ml/confidence.ts` | The low confidence rules. |
| `ml/classes.ts` | Generated from `training/classes.txt` and `class_display_bn.json`. |

## Putting the real models in the app

1. Train and export in Colab. The export step produces
   `plate_detector_int8.tflite` and `char_detector_int8.tflite`.
2. Copy both files into the repo `models/` folder.
3. In `app/`, run `npm run link:models`. It copies them to
   `app/assets/models/` and regenerates `src/ml/modelAssets.ts`. If a file is
   missing the generated value stays `null` and the app keeps using the
   placeholder, so the bundle never breaks.
4. Rebuild the app. `metro.config.js` already treats `.tflite` as an asset.

The model files are in `app/.gitignore` so they are not committed by accident.
To commit them on purpose, use `git add -f app/assets/models/*.tflite` (about
3 MB each, within the 15 MB target). CI builds from git, so a CI built APK only
contains the models if they are committed.

If the class list changes, run `npm run gen:classes`. `npm run check:classes`
fails when `src/ml/classes.ts` is out of date, and CI runs it.

## Verified on the real models (integrate-models branch)

Both files were inspected with Python ai-edge-litert. They were exported by
Ultralytics 8.4.202 through the LiteRT exporter, so they differ from older
TFLite exports in one important way: the input is planar.

| | Plate detector | Character detector |
|---|---|---|
| Input shape | `[1, 3, 640, 640]` NCHW | `[1, 3, 416, 416]` NCHW |
| Input dtype | float32, no quantization (scale 0, zero point 0) | same |
| Output shape | `[1, 5, 8400]` | `[1, 106, 3549]` |
| Output dtype | float32, no quantization | same |
| Box values | normalized 0..1, center x, y, width, height | same |
| Scores | already sigmoid, 0..1, no objectness | same |
| File size | 3.06 MB | 3.08 MB |

Weights are int8 inside, but the graph quantizes and dequantizes at the edges,
so the app sees plain float32 and needs no `inputQuant` or `outputQuant`.

What changed in the app because of this:

* `plateReader.ts` now reorders the letterboxed HWC tensor to CHW
  (`hwcToChw` in `imageOps.ts`) when the input shape is `[1, 3, H, W]`. Older
  NHWC models still work.
* `plateLowScore` went from 0.5 to 0.3. The plate detector's best score on
  four sample photos was 0.5020 every time (the score range is coarse), so a
  0.5 limit would flag nearly every read.
* `decodeYolo` auto detection picks channels first and normalized boxes, which
  matches the real tensors. No change was needed.

Numerical check: `__tests__/ml/realModelOutput.test.ts` decodes raw outputs
saved from the real models for two public dataset photos
(`__tests__/fixtures/real_model_outputs.json`, only anchors scoring above 0.1).
The TS decode, NMS and post-processing give `ঢাকা মেট্রো-গ ৩৯-১৪৭৩` and
`রাজ মেট্রো-ব ১১-০১৫৭`, which match what is printed on the plates. The Python
reference also showed plate boxes close to the dataset labels on four photos.
The character model often outputs many duplicate boxes, NMS removes them.

Still not verified: the JPEG decode and letterbox in JavaScript against the
Python resize, fast-tflite loading these LiteRT files, speed, and accuracy over
a held out set.

## Expected model shapes (before the models existed)

These are the shapes Ultralytics YOLO11n TFLite exports usually have.
**Verify them against the real files** (see the checklist below).

| | Plate detector | Character detector |
|---|---|---|
| Input | `[1, 640, 640, 3]` NHWC, RGB, 0 to 1 | `[1, 416, 416, 3]` NHWC, RGB, 0 to 1 |
| Output | `[1, 5, 8400]` | `[1, 106, 3549]` |
| Output meaning | 4 box values + 1 class score, per anchor | 4 box values + 102 class scores, per anchor |

8400 is the anchor count at 640 (80 x 80 + 40 x 40 + 20 x 20) and 3549 at 416
(52 x 52 + 26 x 26 + 13 x 13).

`decodeYolo` accepts both `[1, 4 + classes, anchors]` and the transposed
`[1, anchors, 4 + classes]`. It tells them apart by which dimension is smaller.
It also accepts boxes either normalized (0 to 1) or in input pixels, and
decides by the largest box value. There is no separate objectness score in
YOLO11, the box score is the best class score.

Models exported with built in NMS (output like `[1, 300, 6]`) are **not**
handled. The training export script does not request that.

## Input handling

`letterbox` mirrors Ultralytics: keep the aspect ratio, scale to fit, pad with
gray 114 of 255, and normalize to 0..1. Boxes are mapped back with
`unletterboxBox`. The plate crop is padded by 5 percent per side
(`platePadding`), because a tight crop can clip edge characters.

The character stage uses the model-input normalized coordinates directly.
Only relative positions matter for line grouping, and the letterbox scale is
uniform, so this is consistent.

## int8 quantization

react-native-fast-tflite exposes each tensor's `dataType` and `shape` but not
its scale and zero point. The code behaves as follows:

* Float32 input and output (what `export(format="tflite", int8=True)` normally
  produces, with int8 weights inside): nothing to configure.
* int8 or uint8 input: the app quantizes with `inputQuant` from the stage
  config. If it is not set, it assumes the usual Ultralytics values: scale
  `1/255` and zero point `-128` for int8, zero point `0` for uint8.
* int8 or uint8 output: `outputQuant` **must** be set in the stage config, or
  the pipeline throws an error that names the missing parameters. Dequantizing
  is `real = scale * (q - zeroPoint)`.

To read the real values, run this once in Colab on the exported file:

```python
import tensorflow as tf
it = tf.lite.Interpreter(model_path="plate_detector_int8.tflite")
it.allocate_tensors()
print(it.get_input_details()[0])    # shape, dtype, quantization
print(it.get_output_details()[0])   # shape, dtype, quantization
```

Copy `quantization` (scale, zero point) into `DEFAULT_PIPELINE_CONFIG` in
`ml/plateReader.ts` if the dtype is not float32.

## Image decode and resize without worklets

The earlier attempt used `react-native-worklets-core` and
`vision-camera-resize-plugin`. `librnworklets.so` crashed on the Galaxy A12
(Android 13) with `cannot locate symbol "__cxa_init_primary_exception"`. This
version avoids both:

* The photo is decoded in JavaScript with `jpeg-js` (pure JS, no native code).
* Letterbox, crop and orientation are plain typed array loops.
* Decoding cost grows with image size, so the sources are limited to about
  1600 px on the long side: the gallery picker uses `maxWidth` and `maxHeight`,
  and the camera screen asks for a photo format near 1600 x 1200. Anything over
  4 million pixels is rejected before decoding.
* Only JPEG is supported. A PNG or WebP gives "Only JPEG images are supported".

Expected cost on a low end phone is a few hundred milliseconds for the decode
plus the model time. This has not been measured.

## Checklist when the real models arrive

1. Print the interpreter details (above) for both files. Confirm input shape,
   dtype, output shape and dtype. Update `inputSize` if the export used a
   different size than 640 and 416.
2. Check the output layout. If the first run throws `Expected N classes...` or
   returns nonsense, compare the shape to the table. `decodeYolo` takes an
   explicit `layout` and `coords` option when auto detection is wrong.
3. Check the box coordinates. Run one known photo and log a few decoded boxes.
   Values below about 1 mean normalized, values in the hundreds mean pixels.
4. Check the resize mode. The training uses Ultralytics letterbox, and the app
   letterboxes the same way. If the exported model was trained or exported
   with stretching instead, boxes will be skewed. Compare boxes on a wide
   photo to find out.
5. Check the color order. The tensor is RGB. Ultralytics models expect RGB.
6. Check NMS settings (`iouThreshold`, `agnosticNms`) on real plates. Agnostic
   NMS is on for the character stage so one glyph cannot give two characters
   of different classes. If adjacent characters with tight spacing get
   merged, raise `iouThreshold` to 0.6 or 0.7. A higher value suppresses less.
7. Check EXIF handling on a photo taken in portrait with the phone camera and
   on a gallery photo. The picker may bake the rotation into the pixels and
   leave the EXIF tag, which would rotate it twice. The orientation code is
   unit tested, but the picker behavior has to be seen on the device.
8. Measure the time for a photo on the Galaxy A12: decode, stage 1, stage 2.
   Then try the `nnapi` and `android-gpu` delegates in `modelLoader.ts`. They
   can be slower or fail for int8 models, so compare accuracy as well.
9. Tune the confidence rules in `ml/confidence.ts` (score threshold 0.6, five
   or six digits) on the held out sets.

## Known gaps

* Not run against real models or on a device.
* The plate crop is taken from the decoded JPEG, which is at most about 1600 px.
  A plate far from the camera may have too few pixels. Check recall on small
  plates.
* The camera format filter (`useCameraFormat`) is untested on the device.

## Patch to react-native-fast-tflite

`app/patches/react-native-fast-tflite+3.0.1.patch` is applied by `patch-package` in the
`postinstall` script, on every `npm install` and `npm ci`, including CI. It changes three
things in the library. Check them again before upgrading the library, and delete the patch
if a new version fixes them.

1. **Release builds could not load the models.** React Native turns a bundled model into a
   bare Android resource name in release builds. The library read it with `java.net.URL`,
   which threw `MalformedURLException: no protocol`. The patched `HybridAssetLoader.kt`
   reads a name without `://` from the app's raw resources, and still reads real URLs.
2. **LiteRT 1.4.0 failed with `Input tensor ... lacks data`** on the exported models. The
   patch moves the Android LiteRT dependency to 1.4.2 in `android/build.gradle`.
3. **Single thread.** The library never set a thread count, so TFLite used one CPU thread.
   `HybridTfliteModule.cpp` now sets up to 4 threads.

The app also reads raw files through the same native `AssetLoader`
(`readFile` in `app/src/ml/imageLoader.ts`), so it depends on item 1 staying in place.
