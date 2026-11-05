import { decodeYolo, iou, nms } from '../../src/ml/yolo';
import type { Detection } from '../../src/ml/postprocess';

type Row = number[]; // cx, cy, w, h, then one score per class

/** Build a [1, 4 + nc, N] tensor from per anchor rows. */
function channelsFirst(rows: Row[], anchors: number): Float32Array {
  const channels = rows[0].length;
  const t = new Float32Array(channels * anchors);
  rows.forEach((row, a) => row.forEach((v, c) => (t[c * anchors + a] = v)));
  return t;
}

/** Build a [1, N, 4 + nc] tensor from per anchor rows. */
function channelsLast(rows: Row[], anchors: number): Float32Array {
  const channels = rows[0].length;
  const t = new Float32Array(channels * anchors);
  rows.forEach((row, a) => row.forEach((v, c) => (t[a * channels + c] = v)));
  return t;
}

const opts = { inputSize: 640, scoreThreshold: 0.25 };

test('decodes a channels first tensor with normalized boxes', () => {
  const rows: Row[] = [
    [0.5, 0.5, 0.2, 0.1, 0.9], // kept
    [0.1, 0.1, 0.1, 0.1, 0.1], // below threshold
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
  ];
  const out = decodeYolo(channelsFirst(rows, 6), [1, 5, 6], opts);
  expect(out.length).toBe(1);
  expect(out[0].cls).toBe(0);
  expect(out[0].x).toBeCloseTo(0.5, 5);
  expect(out[0].y).toBeCloseTo(0.5, 5);
  expect(out[0].w).toBeCloseTo(0.2, 5);
  expect(out[0].h).toBeCloseTo(0.1, 5);
  expect(out[0].conf).toBeCloseTo(0.9, 5);
});

test('decodes a channels last tensor', () => {
  const rows: Row[] = [
    [0.4, 0.6, 0.2, 0.3, 0.8],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
  ];
  const out = decodeYolo(channelsLast(rows, 6), [1, 6, 5], opts);
  expect(out.length).toBe(1);
  expect(out[0].x).toBeCloseTo(0.4, 5);
  expect(out[0].y).toBeCloseTo(0.6, 5);
});

test('pixel coordinates are normalized by the input size', () => {
  const rows: Row[] = [
    [320, 160, 64, 32, 0.9],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
  ];
  const out = decodeYolo(channelsFirst(rows, 6), [1, 5, 6], opts);
  expect(out[0].x).toBeCloseTo(0.5, 5);
  expect(out[0].y).toBeCloseTo(0.25, 5);
  expect(out[0].w).toBeCloseTo(0.1, 5);
  expect(out[0].h).toBeCloseTo(0.05, 5);
});

test('the best class is chosen for each anchor', () => {
  // 3 classes: scores 0.1, 0.7, 0.3 give class 1.
  const rows: Row[] = Array.from({ length: 10 }, () => [0, 0, 0, 0, 0, 0, 0]);
  rows[0] = [0.5, 0.5, 0.1, 0.1, 0.1, 0.7, 0.3];
  const out = decodeYolo(channelsFirst(rows, 10), [1, 7, 10], opts);
  expect(out.length).toBe(1);
  expect(out[0].cls).toBe(1);
  expect(out[0].conf).toBeCloseTo(0.7, 5);
});

test('a class count mismatch is an error', () => {
  const t = new Float32Array(5 * 6);
  expect(() =>
    decodeYolo(t, [1, 5, 6], { ...opts, numClasses: 102 }),
  ).toThrow(/102 classes/);
});

test('a wrong shape is an error', () => {
  expect(() => decodeYolo(new Float32Array(4), [4], opts)).toThrow(
    /Unexpected output shape/,
  );
});

const box = (cls: number, x: number, y: number, conf: number, w = 0.2, h = 0.2): Detection => ({
  cls,
  x,
  y,
  w,
  h,
  conf,
});

test('iou of identical and disjoint boxes', () => {
  expect(iou(box(0, 0.5, 0.5, 1), box(0, 0.5, 0.5, 1))).toBeCloseTo(1, 6);
  expect(iou(box(0, 0.2, 0.2, 1), box(0, 0.8, 0.8, 1))).toBe(0);
});

test('iou of half overlapping boxes is one third', () => {
  expect(iou(box(0, 0.5, 0.5, 1), box(0, 0.6, 0.5, 1))).toBeCloseTo(1 / 3, 5);
});

test('nms keeps the higher score of overlapping boxes of one class', () => {
  const out = nms([box(0, 0.5, 0.5, 0.6), box(0, 0.51, 0.5, 0.9)], { iouThreshold: 0.5 });
  expect(out.length).toBe(1);
  expect(out[0].conf).toBe(0.9);
});

test('nms keeps boxes of different classes unless agnostic', () => {
  const dets = [box(0, 0.5, 0.5, 0.9), box(1, 0.5, 0.5, 0.8)];
  expect(nms(dets, { iouThreshold: 0.5 }).length).toBe(2);
  const agnostic = nms(dets, { iouThreshold: 0.5, agnostic: true });
  expect(agnostic.length).toBe(1);
  expect(agnostic[0].cls).toBe(0);
});

test('nms keeps disjoint boxes and orders by score', () => {
  const out = nms([box(0, 0.2, 0.2, 0.5), box(0, 0.8, 0.8, 0.9)], { iouThreshold: 0.5 });
  expect(out.map(d => d.conf)).toEqual([0.9, 0.5]);
});

test('nms respects maxDetections', () => {
  const dets = [0.1, 0.3, 0.5, 0.7, 0.9].map((x, i) => box(0, x, 0.5, 0.9 - i * 0.1, 0.05, 0.05));
  expect(nms(dets, { iouThreshold: 0.5, maxDetections: 2 }).length).toBe(2);
});

test('nms of nothing is nothing', () => {
  expect(nms([], { iouThreshold: 0.5 })).toEqual([]);
});
