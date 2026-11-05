import { assessConfidence, isLowConfidence } from '../../src/ml/confidence';

const good = ['Dhaka', 'Metro', 'Ga', '2', '7', '8', '4', '7', '6'];
const scores = (n: number, v: number = 0.9) => new Array(n).fill(v);

test('a clean two line plate is not flagged', () => {
  expect(isLowConfidence(good, scores(good.length))).toBe(false);
});

test('a district plate without Metro is not flagged', () => {
  const t = ['Khulna', 'Ta', '1', '1', '2', '3', '4', '5'];
  expect(isLowConfidence(t, scores(t.length))).toBe(false);
});

test('nothing detected is flagged', () => {
  const r = assessConfidence([], []);
  expect(r.low).toBe(true);
  expect(r.reasons).toEqual(['no characters found']);
});

test('one low character score is flagged', () => {
  const s = scores(good.length);
  s[4] = 0.3;
  expect(isLowConfidence(good, s)).toBe(true);
});

test('a score exactly at the threshold passes', () => {
  expect(isLowConfidence(good, scores(good.length, 0.6))).toBe(false);
});

test('a NaN score is treated as low', () => {
  const s = scores(good.length);
  s[0] = NaN;
  expect(isLowConfidence(good, s)).toBe(true);
});

test('too few digits is flagged', () => {
  const t = ['Dhaka', 'Metro', 'Ga', '2', '7', '8', '4'];
  expect(isLowConfidence(t, scores(t.length))).toBe(true);
});

test('too many digits is flagged', () => {
  const t = [...good, '9'];
  expect(isLowConfidence(t, scores(t.length))).toBe(true);
});

test('missing series letter is flagged', () => {
  const t = ['Dhaka', 'Metro', '2', '7', '8', '4', '7', '6'];
  expect(isLowConfidence(t, scores(t.length))).toBe(true);
});

test('two series letters are flagged', () => {
  const t = ['Dhaka', 'Metro', 'Ga', 'Gha', '2', '7', '8', '4', '7', '6'];
  expect(isLowConfidence(t, scores(t.length))).toBe(true);
});

test('missing district is flagged', () => {
  const t = ['Metro', 'Ga', '2', '7', '8', '4', '7', '6'];
  expect(isLowConfidence(t, scores(t.length))).toBe(true);
});

test('unknown class is flagged', () => {
  const t = ['Dhaka', 'Metro', 'Ga', '?', '7', '8', '4', '7', '6'];
  expect(isLowConfidence(t, scores(t.length))).toBe(true);
});

test('mismatched scores length is flagged', () => {
  expect(isLowConfidence(good, scores(3))).toBe(true);
});

test('options can loosen the digit range', () => {
  const t = ['Dhaka', 'Ga', '2', '7', '8', '4'];
  expect(isLowConfidence(t, scores(t.length))).toBe(true);
  expect(isLowConfidence(t, scores(t.length), { minDigits: 4 })).toBe(false);
});
