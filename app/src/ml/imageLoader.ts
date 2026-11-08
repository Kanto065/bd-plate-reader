/**
 * Loads an image file into raw RGB pixels without any native resize plugin.
 *
 * Reads the file bytes, decodes the JPEG in JavaScript with jpeg-js, then
 * applies the EXIF orientation. JS decoding is slow for big photos, so the
 * image must be small. The picker and the camera are asked for about 1600 px
 * on the long side (see MAX_SIDE), and anything above MAX_PIXELS is rejected
 * before decoding instead of freezing the app.
 */
import { NativeModules } from 'react-native';
import jpeg from 'jpeg-js';
import { timed, timedAsync } from './timing';
import {
  ImageSource,
  PixelBox,
  RgbImage,
  applyExifOrientation,
  isJpeg,
  readExifOrientation,
  readJpegSize,
  rgbSource,
} from './imageOps';

/** Long side, in pixels, requested from the camera and the gallery picker. */
export const MAX_SIDE = 1600;
/** Hard limit on the decoded size. 1600 x 1200 is 1.9 MP. */
export const MAX_PIXELS = 4_000_000;

export class ImageLoadError extends Error {}

/** Decode JPEG bytes into an upright RGB image. Pure, used by the tests. */
export function decodeJpegBytes(bytes: Uint8Array): RgbImage {
  if (!isJpeg(bytes)) {
    throw new ImageLoadError('Only JPEG images are supported');
  }
  const size = readJpegSize(bytes);
  if (size && size.width * size.height > MAX_PIXELS) {
    throw new ImageLoadError(
      'Image is too large (' + size.width + ' x ' + size.height + ')',
    );
  }
  const decoded = timed('jpegDecode', () =>
    jpeg.decode(bytes, {
      useTArray: true,
      formatAsRGBA: false,
      maxMemoryUsageInMB: 256,
    }),
  );
  const img: RgbImage = {
    data: decoded.data,
    width: decoded.width,
    height: decoded.height,
  };
  return timed('exifRotate', () =>
    applyExifOrientation(img, readExifOrientation(bytes)),
  );
}

/**
 * Long side of the photo the native module decodes. The plate crop is cut from
 * this, so it is larger than the 640 px the plate detector sees.
 */
export const NATIVE_MAX_SIDE = 2048;

interface NativeLetterboxed {
  srcWidth: number;
  srcHeight: number;
  scale: number;
  padX: number;
  padY: number;
  path: string; // file with the planar float32 model input
}

/** Android only. Read lazily so tests can install a fake. */
function nativeModule() {
  return NativeModules.PlateImage as
    | {
        letterbox(
          uri: string,
          maxSide: number,
          size: number,
          crop: number[] | null,
        ): Promise<NativeLetterboxed>;
      }
    | undefined;
}

let fileReader: { loadAsset(path: string): Promise<ArrayBuffer> } | undefined;

/** Raw bytes of a file, from the native loader that react-native-fast-tflite ships. */
function readFile(path: string): Promise<ArrayBuffer> {
  if (!fileReader) {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const { NitroModules } = require('react-native-nitro-modules');
    fileReader = NitroModules.createHybridObject('AssetLoader');
  }
  return fileReader!.loadAsset(path);
}

function nativeSource(
  native: NonNullable<ReturnType<typeof nativeModule>>,
  uri: string,
  crop: PixelBox | null,
): ImageSource {
  return {
    letterbox: async size => {
      const r = await timedAsync('nativeLetterbox', () =>
        native.letterbox(
          uri,
          NATIVE_MAX_SIDE,
          size,
          crop && [crop.x1, crop.y1, crop.x2, crop.y2],
        ),
      );
      const buffer = await timedAsync('readTensor', () => readFile(r.path));
      return {
        tensor: new Float32Array(buffer),
        size,
        planar: true,
        scale: r.scale,
        padX: r.padX,
        padY: r.padY,
        srcWidth: r.srcWidth,
        srcHeight: r.srcHeight,
      };
    },
    // The crop box is in pixels of the photo the native module decoded, which
    // is the srcWidth x srcHeight it reported for the full photo.
    crop: box => nativeSource(native, uri, box),
  };
}

/** A photo ready for the pipeline: native on Android, JavaScript elsewhere. */
export async function loadImageSource(uri: string): Promise<ImageSource> {
  const native = nativeModule();
  return native ? nativeSource(native, uri, null) : rgbSource(await loadRgbImage(uri));
}

export async function loadRgbImage(uri: string): Promise<RgbImage> {
  const buffer = await timedAsync('fileRead', async () => {
    const response = await fetch(uri);
    return response.arrayBuffer();
  });
  return decodeJpegBytes(new Uint8Array(buffer));
}
