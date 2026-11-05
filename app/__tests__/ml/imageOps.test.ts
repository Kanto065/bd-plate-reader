import {
  LETTERBOX_PAD,
  RgbImage,
  applyExifOrientation,
  cropRgb,
  isJpeg,
  letterbox,
  padBox,
  readExifOrientation,
  readJpegSize,
  unletterboxBox,
} from '../../src/ml/imageOps';

/** Image where the red channel of pixel (x, y) is x + y * width. */
function indexImage(width: number, height: number): RgbImage {
  const data = new Uint8Array(width * height * 3);
  for (let i = 0; i < width * height; i++) {
    data[i * 3] = i;
    data[i * 3 + 1] = 0;
    data[i * 3 + 2] = 0;
  }
  return { data, width, height };
}

function redRows(img: RgbImage): number[][] {
  const rows: number[][] = [];
  for (let y = 0; y < img.height; y++) {
    const row: number[] = [];
    for (let x = 0; x < img.width; x++) {
      row.push(img.data[(y * img.width + x) * 3]);
    }
    rows.push(row);
  }
  return rows;
}

// Source rows are [0 1 2] and [3 4 5].
const ORIENTATION_RESULTS: Record<number, number[][]> = {
  2: [[2, 1, 0], [5, 4, 3]],
  3: [[5, 4, 3], [2, 1, 0]],
  4: [[3, 4, 5], [0, 1, 2]],
  5: [[0, 3], [1, 4], [2, 5]],
  6: [[3, 0], [4, 1], [5, 2]],
  7: [[5, 2], [4, 1], [3, 0]],
  8: [[2, 5], [1, 4], [0, 3]],
};

test.each(Object.keys(ORIENTATION_RESULTS).map(Number))(
  'EXIF orientation %i',
  orientation => {
    const out = applyExifOrientation(indexImage(3, 2), orientation);
    expect(redRows(out)).toEqual(ORIENTATION_RESULTS[orientation]);
  },
);

test('EXIF orientation 1 and invalid values return the same image', () => {
  const img = indexImage(3, 2);
  expect(applyExifOrientation(img, 1)).toBe(img);
  expect(applyExifOrientation(img, 0)).toBe(img);
  expect(applyExifOrientation(img, 9)).toBe(img);
});

/** Minimal JPEG like byte stream with an EXIF orientation tag. */
function jpegWithExif(orientation: number, little: boolean): Uint8Array {
  const u16 = (v: number) => (little ? [v & 255, v >> 8] : [v >> 8, v & 255]);
  const u32 = (v: number) =>
    little ? [v & 255, (v >> 8) & 255, 0, 0] : [0, 0, (v >> 8) & 255, v & 255];
  const tiff = [
    ...(little ? [0x49, 0x49] : [0x4d, 0x4d]),
    ...u16(42),
    ...u32(8),
    ...u16(1), // one entry
    ...u16(0x0112),
    ...u16(3),
    ...u32(1),
    ...u16(orientation),
    0,
    0,
    ...u32(0),
  ];
  const payload = [0x45, 0x78, 0x69, 0x66, 0, 0, ...tiff];
  const len = payload.length + 2;
  return Uint8Array.from([
    0xff, 0xd8, 0xff, 0xe1, len >> 8, len & 255, ...payload, 0xff, 0xd9,
  ]);
}

test.each([1, 3, 6, 8])('reads EXIF orientation %i, both byte orders', o => {
  expect(readExifOrientation(jpegWithExif(o, false))).toBe(o);
  expect(readExifOrientation(jpegWithExif(o, true))).toBe(o);
});

test('missing EXIF or non JPEG bytes give orientation 1', () => {
  expect(readExifOrientation(Uint8Array.from([0xff, 0xd8, 0xff, 0xd9]))).toBe(1);
  expect(readExifOrientation(Uint8Array.from([1, 2, 3, 4, 5, 6]))).toBe(1);
  expect(readExifOrientation(new Uint8Array(0))).toBe(1);
});

test('isJpeg checks the signature', () => {
  expect(isJpeg(Uint8Array.from([0xff, 0xd8, 0xff, 0xe0]))).toBe(true);
  expect(isJpeg(Uint8Array.from([0x89, 0x50, 0x4e, 0x47]))).toBe(false);
});

test('reads the JPEG size from the SOF segment', () => {
  const bytes = Uint8Array.from([
    0xff, 0xd8,
    0xff, 0xc0, 0x00, 0x11, 0x08, 0x04, 0xb0, 0x06, 0x40, 0x03, 1, 0x22, 0, 2, 0x11, 1, 3, 0x11, 1,
    0xff, 0xd9,
  ]);
  expect(readJpegSize(bytes)).toEqual({ width: 1600, height: 1200 });
  expect(readJpegSize(Uint8Array.from([1, 2, 3]))).toBeNull();
});

test('crop copies the requested pixels and clamps to the image', () => {
  const img = indexImage(4, 3);
  const c = cropRgb(img, { x1: 1, y1: 1, x2: 3, y2: 3 });
  expect(c.width).toBe(2);
  expect(c.height).toBe(2);
  expect(redRows(c)).toEqual([[5, 6], [9, 10]]);
  const edge = cropRgb(img, { x1: -5, y1: -5, x2: 100, y2: 100 });
  expect(edge.width).toBe(4);
  expect(edge.height).toBe(3);
});

test('crop of an empty box still returns one pixel', () => {
  const c = cropRgb(indexImage(4, 3), { x1: 2, y1: 2, x2: 2, y2: 2 });
  expect(c.width).toBe(1);
  expect(c.height).toBe(1);
});

test('letterbox pads a wide image on top and bottom', () => {
  const img = indexImage(4, 2);
  const lb = letterbox(img, 4);
  expect(lb.scale).toBe(1);
  expect(lb.padX).toBe(0);
  expect(lb.padY).toBe(1);
  expect(lb.tensor.length).toBe(4 * 4 * 3);
  // Row 0 is padding, row 1 is image row 0, row 2 is image row 1, row 3 padding.
  expect(lb.tensor[0]).toBeCloseTo(LETTERBOX_PAD, 6);
  expect(lb.tensor[(1 * 4 + 0) * 3]).toBeCloseTo(0 / 255, 6);
  expect(lb.tensor[(1 * 4 + 3) * 3]).toBeCloseTo(3 / 255, 6);
  expect(lb.tensor[(2 * 4 + 2) * 3]).toBeCloseTo(6 / 255, 6);
  expect(lb.tensor[(3 * 4 + 1) * 3]).toBeCloseTo(LETTERBOX_PAD, 6);
});

test('letterbox values stay inside 0..1 when downscaling', () => {
  const lb = letterbox(indexImage(8, 4), 4);
  expect(lb.scale).toBe(0.5);
  for (const v of lb.tensor) {
    expect(v).toBeGreaterThanOrEqual(0);
    expect(v).toBeLessThanOrEqual(1);
  }
});

test('unletterbox maps a full image box back to the whole image', () => {
  const lb = letterbox(indexImage(4, 2), 4);
  const b = unletterboxBox(lb, 0.5, 0.5, 1, 0.5);
  expect(b).toEqual({ x1: 0, y1: 0, x2: 4, y2: 2 });
});

test('unletterbox accounts for scale', () => {
  const lb = letterbox(indexImage(8, 4), 4); // scale 0.5, padY 1
  const b = unletterboxBox(lb, 0.5, 0.5, 0.5, 0.25);
  expect(b).toEqual({ x1: 2, y1: 1, x2: 6, y2: 3 });
});

test('padBox grows a box and clamps it to the image', () => {
  const b = padBox({ x1: 10, y1: 10, x2: 30, y2: 20 }, 0.1, 100, 100);
  expect(b).toEqual({ x1: 8, y1: 9, x2: 32, y2: 21 });
  const edge = padBox({ x1: 1, y1: 1, x2: 99, y2: 99 }, 0.5, 100, 100);
  expect(edge).toEqual({ x1: 0, y1: 0, x2: 100, y2: 100 });
});
