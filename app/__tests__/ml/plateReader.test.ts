import { CLASS_NAMES } from '../../src/ml/classes';
import { ModelHandle, runPipeline } from '../../src/ml/plateReader';
import { quantize } from '../../src/ml/quant';
import type { RgbImage } from '../../src/ml/imageOps';

const ID: Record<string, number> = {};
CLASS_NAMES.forEach((n, i) => {
  ID[n] = i;
});

type Row = { x: number; y: number; w: number; h: number; scores: Record<number, number> };

/** Build a channels first [1, 4 + nc, anchors] tensor. */
function makeOutput(rows: Row[], nc: number, anchors: number): Float32Array {
  const channels = 4 + nc;
  const t = new Float32Array(channels * anchors);
  rows.forEach((r, a) => {
    t[0 * anchors + a] = r.x;
    t[1 * anchors + a] = r.y;
    t[2 * anchors + a] = r.w;
    t[3 * anchors + a] = r.h;
    for (const [cls, score] of Object.entries(r.scores)) {
      t[(4 + Number(cls)) * anchors + a] = score;
    }
  });
  return t;
}

const f32 = (t: Float32Array) =>
  t.buffer.slice(t.byteOffset, t.byteOffset + t.byteLength) as ArrayBuffer;

function fakeModel(
  output: Float32Array,
  shape: number[],
  inputType = 'float32',
  outputType = 'float32',
  rawOutput?: ArrayBuffer,
): ModelHandle & { lastInput?: ArrayBuffer } {
  const model: ModelHandle & { lastInput?: ArrayBuffer } = {
    inputs: [{ dataType: inputType, shape: [1, 0, 0, 3] }],
    outputs: [{ dataType: outputType, shape }],
    run: async input => {
      model.lastInput = input[0];
      return [rawOutput ?? f32(output)];
    },
  };
  return model;
}

const PLATE_ANCHORS = 20;
const CHAR_ANCHORS = 200; // more anchors than channels (106), so channels first

function plateOutput(conf: number): Float32Array {
  return makeOutput(
    [{ x: 0.5, y: 0.5, w: 0.6, h: 0.3, scores: { 0: conf } }],
    1,
    PLATE_ANCHORS,
  );
}

function charRows(overrides: Record<number, number> = {}): Row[] {
  const rows: Row[] = [];
  const add = (name: string, x: number, y: number, w = 0.06) => {
    const i = rows.length;
    const conf = overrides[i] ?? 0.9;
    rows.push({ x, y, w, h: 0.2, scores: { [ID[name]]: conf } });
  };
  add('Dhaka', 0.25, 0.28, 0.2);
  add('Metro', 0.55, 0.28, 0.2);
  add('Ga', 0.8, 0.28);
  '278476'.split('').forEach((d, i) => add(d, 0.12 + 0.14 * i, 0.72));
  return rows;
}

function charOutput(rows: Row[]): Float32Array {
  return makeOutput(rows, 102, CHAR_ANCHORS);
}

function grayImage(width: number, height: number): RgbImage {
  return { data: new Uint8Array(width * height * 3).fill(128), width, height };
}

const plateShape = [1, 5, PLATE_ANCHORS];
const charShape = [1, 106, CHAR_ANCHORS];

test('reads a two line plate end to end with float models', async () => {
  const plate = fakeModel(plateOutput(0.9), plateShape);
  const chars = fakeModel(charOutput(charRows()), charShape);
  const result = await runPipeline({ plate, chars }, grayImage(640, 480));
  expect(result.plateFound).toBe(true);
  expect(result.text).toBe('ঢাকা মেট্রো-গ ২৭-৮৪৭৬');
  expect(result.lowConfidence).toBe(false);
  expect(result.reasons).toEqual([]);
  // The plate model got a 640 x 640 x 3 float32 tensor, the char model 416.
  expect(plate.lastInput!.byteLength).toBe(640 * 640 * 3 * 4);
  expect(chars.lastInput!.byteLength).toBe(416 * 416 * 3 * 4);
});

test('no plate found gives empty text and low confidence', async () => {
  const plate = fakeModel(new Float32Array(5 * PLATE_ANCHORS), plateShape);
  const chars = fakeModel(charOutput(charRows()), charShape);
  const result = await runPipeline({ plate, chars }, grayImage(640, 480));
  expect(result.plateFound).toBe(false);
  expect(result.text).toBe('');
  expect(result.lowConfidence).toBe(true);
  expect(chars.lastInput).toBeUndefined();
});

test('a low scoring character flags low confidence but keeps the text', async () => {
  const plate = fakeModel(plateOutput(0.9), plateShape);
  // Character 4 scores 0.3, above the 0.25 keep threshold, below 0.6.
  const chars = fakeModel(charOutput(charRows({ 4: 0.3 })), charShape);
  const result = await runPipeline({ plate, chars }, grayImage(640, 480));
  expect(result.text).toBe('ঢাকা মেট্রো-গ ২৭-৮৪৭৬');
  expect(result.lowConfidence).toBe(true);
});

test('a dropped character changes the digit count and flags low confidence', async () => {
  const plate = fakeModel(plateOutput(0.9), plateShape);
  // Characters 4 and 5 score 0.2, below the 0.25 keep threshold, so both are
  // removed and only four digits are left.
  const chars = fakeModel(charOutput(charRows({ 4: 0.2, 5: 0.2 })), charShape);
  const result = await runPipeline({ plate, chars }, grayImage(640, 480));
  expect(result.text).toBe('ঢাকা মেট্রো-গ ২৪৭৬');
  expect(result.lowConfidence).toBe(true);
});

test('an unsure plate box flags low confidence', async () => {
  const plate = fakeModel(plateOutput(0.28), plateShape);
  const chars = fakeModel(charOutput(charRows()), charShape);
  const result = await runPipeline({ plate, chars }, grayImage(640, 480));
  expect(result.plateFound).toBe(true);
  expect(result.lowConfidence).toBe(true);
  expect(result.reasons).toContain('plate detection score is low');
});

test('int8 models are quantized on input and dequantized on output', async () => {
  const q = { scale: 1 / 127, zeroPoint: 0 };
  const plateF = plateOutput(0.9);
  const charF = charOutput(charRows());
  const plateRaw = quantize(plateF, q, 'int8').buffer as ArrayBuffer;
  const charRaw = quantize(charF, q, 'int8').buffer as ArrayBuffer;
  const plate = fakeModel(plateF, plateShape, 'int8', 'int8', plateRaw);
  const chars = fakeModel(charF, charShape, 'int8', 'int8', charRaw);
  const cfg = {
    plate: {
      inputSize: 640, scoreThreshold: 0.25, iouThreshold: 0.5, maxDetections: 5,
      agnosticNms: true, outputQuant: q,
    },
    chars: {
      inputSize: 416, scoreThreshold: 0.25, iouThreshold: 0.5, maxDetections: 20,
      agnosticNms: true, outputQuant: q,
    },
    platePadding: 0.05,
    charMinConf: 0.25,
    plateLowScore: 0.5,
  };
  const result = await runPipeline({ plate, chars }, grayImage(640, 480), cfg);
  expect(result.text).toBe('ঢাকা মেট্রো-গ ২৭-৮৪৭৬');
  expect(plate.lastInput!.byteLength).toBe(640 * 640 * 3); // one byte per value
  expect(new Int8Array(plate.lastInput!)[0]).toBeGreaterThan(-128);
});

test('an int8 output without quantization parameters is an error', async () => {
  const raw = new Int8Array(5 * PLATE_ANCHORS).buffer as ArrayBuffer;
  const plate = fakeModel(new Float32Array(0), plateShape, 'float32', 'int8', raw);
  const chars = fakeModel(charOutput(charRows()), charShape);
  await expect(runPipeline({ plate, chars }, grayImage(64, 48))).rejects.toThrow(
    /quantization/,
  );
});

test('crops the plate from the right place in a non square photo', async () => {
  // A 1000 x 500 photo is letterboxed to 640: scale 0.64, padY 160. A plate box
  // centered at (0.5, 0.5) with size 0.6 x 0.3 is the pixel box x 200..800,
  // y 100..400. The crop adds 5 percent on each side: x 170..830, y 85..415.
  const w = 1000;
  const h = 500;
  const data = new Uint8Array(w * h * 3);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const inside = x >= 200 && x < 800 && y >= 100 && y < 400;
      const i = (y * w + x) * 3;
      data[i] = inside ? 255 : 0; // red inside the plate
      data[i + 2] = inside ? 0 : 255; // blue outside
    }
  }
  const plate = fakeModel(plateOutput(0.9), plateShape);
  const chars = fakeModel(charOutput(charRows()), charShape);
  await runPipeline({ plate, chars }, { data, width: w, height: h });
  const t = new Float32Array(chars.lastInput!);
  const px = (x: number, y: number) => t[(y * 416 + x) * 3]; // red channel
  // The 660 x 330 crop is scaled to 416 x 208 with 104 rows of padding on top.
  expect(px(0, 0)).toBeCloseTo(114 / 255, 3); // padding
  expect(px(208, 208)).toBeGreaterThan(0.95); // plate center is red
  expect(px(3, 208)).toBeLessThan(0.05); // left margin is outside the plate, blue
  expect(px(412, 208)).toBeLessThan(0.05); // right margin
});

test('runPipeline accepts an ImageSource and crops with the source dimensions', async () => {
  const { letterbox } = require('../../src/ml/imageOps');
  const calls: string[] = [];
  const source = {
    letterbox: async (size: number) => {
      calls.push('letterbox' + size);
      return { ...letterbox(grayImage(64, 48), size), srcWidth: 1000, srcHeight: 500 };
    },
    crop: () => {
      calls.push('crop');
      return source;
    },
  };
  const plate = fakeModel(plateOutput(0.9), plateShape);
  const chars = fakeModel(charOutput(charRows()), charShape);
  const result = await runPipeline({ plate, chars }, source);
  expect(result.plateFound).toBe(true);
  expect(calls).toEqual(['letterbox640', 'crop', 'letterbox416']);
});
