import {
  bufferToFloat,
  dequantize,
  floatToBuffer,
  quantize,
} from '../../src/ml/quant';

test('int8 round trip is close to the original', () => {
  const q = { scale: 1 / 127, zeroPoint: 0 };
  const values = Float32Array.from([0, 0.25, 0.5, 1, -1]);
  const back = dequantize(quantize(values, q, 'int8') as Int8Array, q);
  values.forEach((v, i) => expect(back[i]).toBeCloseTo(v, 2));
});

test('quantize clamps to the integer range', () => {
  const q = { scale: 0.01, zeroPoint: 0 };
  const i8 = quantize(Float32Array.from([5, -5]), q, 'int8');
  expect(Array.from(i8)).toEqual([127, -128]);
  const u8 = quantize(Float32Array.from([5, -5]), q, 'uint8');
  expect(Array.from(u8)).toEqual([255, 0]);
});

test('zero point shifts the values', () => {
  const q = { scale: 1 / 255, zeroPoint: -128 };
  const out = quantize(Float32Array.from([0, 1]), q, 'int8');
  expect(Array.from(out)).toEqual([-128, 127]);
  expect(Array.from(dequantize(out as Int8Array, q))).toEqual([0, 1]);
});

test('bufferToFloat reads float32 directly', () => {
  const f = Float32Array.from([1.5, 2.5]);
  expect(Array.from(bufferToFloat(f.buffer, 'float32'))).toEqual([1.5, 2.5]);
});

test('bufferToFloat dequantizes int8 and needs parameters', () => {
  const raw = Int8Array.from([-128, 0, 127]);
  const q = { scale: 0.5, zeroPoint: -128 };
  expect(Array.from(bufferToFloat(raw.buffer, 'int8', q))).toEqual([0, 64, 127.5]);
  expect(() => bufferToFloat(raw.buffer, 'int8')).toThrow(/quantization/);
});

test('bufferToFloat rejects unsupported types', () => {
  expect(() => bufferToFloat(new ArrayBuffer(4), 'float16')).toThrow(/Unsupported/);
});

test('floatToBuffer keeps float32 and quantizes integer inputs', () => {
  const f = Float32Array.from([0, 0.5, 1]);
  expect(new Float32Array(floatToBuffer(f, 'float32'))).toEqual(f);
  const i8 = new Int8Array(floatToBuffer(f, 'int8'));
  expect(Array.from(i8)).toEqual([-128, 0, 127]);
  const u8 = new Uint8Array(floatToBuffer(f, 'uint8'));
  expect(Array.from(u8)).toEqual([0, 128, 255]);
});

test('floatToBuffer rejects unsupported types', () => {
  expect(() => floatToBuffer(new Float32Array(1), 'float16')).toThrow(/Unsupported/);
});
