import jpeg from 'jpeg-js';
import { MAX_PIXELS, decodeJpegBytes } from '../../src/ml/imageLoader';

/** Encode a width x height image where the left half is red, right half blue. */
function makeJpeg(width: number, height: number): Uint8Array {
  const data = new Uint8Array(width * height * 4);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = (y * width + x) * 4;
      const left = x < width / 2;
      data[i] = left ? 255 : 0;
      data[i + 1] = 0;
      data[i + 2] = left ? 0 : 255;
      data[i + 3] = 255;
    }
  }
  return new Uint8Array(jpeg.encode({ data, width, height }, 95).data);
}

/** Insert a big endian EXIF orientation segment right after the JPEG signature. */
function withOrientation(bytes: Uint8Array, orientation: number): Uint8Array {
  const tiff = [
    0x4d, 0x4d, 0, 42, 0, 0, 0, 8, 0, 1, 0x01, 0x12, 0, 3, 0, 0, 0, 1,
    0, orientation, 0, 0, 0, 0, 0, 0,
  ];
  const payload = [0x45, 0x78, 0x69, 0x66, 0, 0, ...tiff];
  const len = payload.length + 2;
  const segment = [0xff, 0xe1, len >> 8, len & 255, ...payload];
  const out = new Uint8Array(bytes.length + segment.length);
  out.set(bytes.subarray(0, 2), 0);
  out.set(segment, 2);
  out.set(bytes.subarray(2), 2 + segment.length);
  return out;
}

const isRed = (img: { data: ArrayLike<number> }, x: number, y: number, w: number) =>
  img.data[(y * w + x) * 3] > 200;

test('decodes a JPEG into RGB pixels', () => {
  const img = decodeJpegBytes(makeJpeg(16, 8));
  expect(img.width).toBe(16);
  expect(img.height).toBe(8);
  expect(img.data.length).toBe(16 * 8 * 3);
  expect(isRed(img, 2, 4, 16)).toBe(true);
  expect(isRed(img, 13, 4, 16)).toBe(false);
});

test('applies EXIF orientation 6 (rotate 90 clockwise)', () => {
  const img = decodeJpegBytes(withOrientation(makeJpeg(16, 8), 6));
  expect(img.width).toBe(8);
  expect(img.height).toBe(16);
  // Left half (red) becomes the top half after rotating clockwise.
  expect(isRed(img, 4, 2, 8)).toBe(true);
  expect(isRed(img, 4, 13, 8)).toBe(false);
});

test('rejects data that is not a JPEG', () => {
  expect(() => decodeJpegBytes(Uint8Array.from([0x89, 0x50, 0x4e, 0x47, 0, 0]))).toThrow(
    /JPEG/,
  );
});

test('rejects images above the pixel limit before decoding', () => {
  // Fake a header that claims 4000 x 3000 pixels.
  const w = 4000;
  const h = 3000;
  expect(w * h).toBeGreaterThan(MAX_PIXELS);
  const bytes = Uint8Array.from([
    0xff, 0xd8, 0xff, 0xc0, 0x00, 0x11, 0x08, h >> 8, h & 255, w >> 8, w & 255,
    0x03, 1, 0x22, 0, 2, 0x11, 1, 3, 0x11, 1, 0xff, 0xd9,
  ]);
  expect(() => decodeJpegBytes(bytes)).toThrow(/too large/);
});

describe('native decoder', () => {
  const { NativeModules } = require('react-native');
  afterEach(() => {
    delete NativeModules.PlateImage;
  });

  test('loadRgbImage uses the native decoder and turns base64 into RGB bytes', async () => {
    const { loadRgbImage, NATIVE_MAX_SIDE } = require('../../src/ml/imageLoader');
    // Two pixels: red, then blue.
    const data = '/wAAAAD/'; // bytes 255,0,0,0,0,255
    const decode = jest.fn().mockResolvedValue({ width: 2, height: 1, data });
    NativeModules.PlateImage = { decode };
    const img = await loadRgbImage('file:///photo.jpg');
    expect(decode).toHaveBeenCalledWith('file:///photo.jpg', NATIVE_MAX_SIDE);
    expect(img.width).toBe(2);
    expect(img.height).toBe(1);
    expect(Array.from(img.data)).toEqual([255, 0, 0, 0, 0, 255]);
  });
});
