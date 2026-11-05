/**
 * Decodes raw outputs saved from the real int8 models (run with Python
 * ai-edge-litert on two public dataset photos). The fixture keeps only the
 * anchors whose best class score is above 0.1, everything else is zero.
 * This pins the real layout: float32 [1, 4 + classes, anchors], normalized
 * center boxes.
 */
import fixture from '../fixtures/real_model_outputs.json';
import { CLASS_DISPLAY, CLASS_NAMES } from '../../src/ml/classes';
import { hwcToChw } from '../../src/ml/imageOps';
import { readPlateDetailed } from '../../src/ml/postprocess';
import { decodeYolo, nms } from '../../src/ml/yolo';

type Sparse = { shape: number[]; anchors: [number, number[]][] };

function dense(s: Sparse): Float32Array {
  const [channels, anchors] = s.shape;
  const t = new Float32Array(channels * anchors);
  for (const [a, col] of s.anchors) {
    col.forEach((v, c) => {
      t[c * anchors + a] = v;
    });
  }
  return t;
}

const data = fixture as unknown as Record<string, { plate: Sparse; chars: Sparse }>;

test('plate detector output decodes to one normalized box', () => {
  const s = data.Vehicle1.plate;
  const dets = nms(
    decodeYolo(dense(s), [1, ...s.shape], { inputSize: 640, scoreThreshold: 0.25, numClasses: 1 }),
    { iouThreshold: 0.5, agnostic: true, maxDetections: 5 },
  );
  expect(dets.length).toBeGreaterThanOrEqual(1);
  const b = dets[0];
  expect(b.x).toBeGreaterThan(0.3);
  expect(b.x).toBeLessThan(0.6);
  expect(b.w).toBeGreaterThan(0.1);
  expect(b.w).toBeLessThan(0.4);
  expect(b.conf).toBeGreaterThan(0.3);
});

function read(name: string): string {
  const s = data[name].chars;
  const dets = nms(
    decodeYolo(dense(s), [1, ...s.shape], {
      inputSize: 416,
      scoreThreshold: 0.25,
      numClasses: CLASS_NAMES.length,
    }),
    { iouThreshold: 0.5, agnostic: true, maxDetections: 20 },
  );
  return readPlateDetailed(dets, CLASS_NAMES, CLASS_DISPLAY, 0.25).text;
}

test('single line style plate from the real character model', () => {
  expect(CLASS_NAMES.length).toBe(102);
  const text = read('Vehicle1');
  // eslint-disable-next-line no-console
  console.log('Vehicle1 ->', text);
  expect(text).toContain('ঢাকা');
  expect(text).toContain('১৪৭৩');
});

test('two line plate from the real character model', () => {
  const text = read('Vehicle10');
  // eslint-disable-next-line no-console
  console.log('Vehicle10 ->', text);
  expect(text).toContain('রাজ');
  expect(text).toContain('০১৫৭');
});

test('hwcToChw reorders interleaved pixels to planes', () => {
  // 2 x 2 image, pixel i has rgb (10i, 10i+1, 10i+2)
  const hwc = new Float32Array(12);
  for (let i = 0; i < 4; i++) {
    hwc.set([10 * i, 10 * i + 1, 10 * i + 2], i * 3);
  }
  expect(Array.from(hwcToChw(hwc, 2))).toEqual([
    0, 10, 20, 30, 1, 11, 21, 31, 2, 12, 22, 32,
  ]);
});
