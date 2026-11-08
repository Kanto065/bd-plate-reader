/**
 * Loads an image file into raw RGB pixels without any native resize plugin.
 *
 * Reads the file bytes, decodes the JPEG in JavaScript with jpeg-js, then
 * applies the EXIF orientation. JS decoding is slow for big photos, so the
 * image must be small. The picker and the camera are asked for about 1600 px
 * on the long side (see MAX_SIDE), and anything above MAX_PIXELS is rejected
 * before decoding instead of freezing the app.
 */
import jpeg from 'jpeg-js';
import { timed, timedAsync } from './timing';
import {
  RgbImage,
  applyExifOrientation,
  isJpeg,
  readExifOrientation,
  readJpegSize,
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

export async function loadRgbImage(uri: string): Promise<RgbImage> {
  const buffer = await timedAsync('fileRead', async () => {
    const response = await fetch(uri);
    return response.arrayBuffer();
  });
  return decodeJpegBytes(new Uint8Array(buffer));
}
