import {
  Detection,
  editDistance,
  estimateTilt,
  groupIntoLines,
  median,
  plateTokens,
  readPlateDetailed,
  readPlateText,
  rotateDetections,
} from '../../src/ml/postprocess';
import { CLASS_DISPLAY, CLASS_NAMES } from '../../src/ml/classes';

const ID: Record<string, number> = {};
CLASS_NAMES.forEach((n, i) => {
  ID[n] = i;
});

function det(
  name: string,
  x: number,
  y: number,
  w: number = 0.06,
  h: number = 0.2,
  conf: number = 0.9,
): Detection {
  return { cls: ID[name], x, y, w, h, conf };
}

/** Dhaka Metro Ga on top, 27-8476 below. Detections are deliberately shuffled. */
function twoLinePlate(): Detection[] {
  const top = [
    det('Dhaka', 0.25, 0.28, 0.2),
    det('Metro', 0.55, 0.28, 0.2),
    det('Ga', 0.8, 0.28),
  ];
  const bottom = '278476'.split('').map((d, i) => det(d, 0.12 + 0.14 * i, 0.72));
  const items = [...top, ...bottom];
  return [5, 0, 8, 2, 7, 1, 4, 6, 3].map(i => items[i]);
}

/** Dhaka Metro Gha 112345 on one line. */
function singleLinePlate(): Detection[] {
  const seq = ['Dhaka', 'Metro', 'Gha', '1', '1', '2', '3', '4', '5'];
  return seq.map((n, i) => det(n, 0.05 + 0.1 * i, 0.5 + (i % 2 ? 0.01 : -0.01)));
}

const names = (line: Detection[]) => line.map(d => CLASS_NAMES[d.cls]);

test('class names are loaded', () => {
  expect(CLASS_NAMES.length).toBe(102);
  expect(CLASS_NAMES[0]).toBe('0');
  expect(CLASS_NAMES[10]).toBe('Metro');
});

test('two lines are grouped top to bottom', () => {
  const lines = groupIntoLines(twoLinePlate());
  expect(lines.length).toBe(2);
  expect(names(lines[0])).toEqual(['Dhaka', 'Metro', 'Ga']);
  expect(names(lines[1])).toEqual('278476'.split(''));
});

test('a single line is one group sorted by x', () => {
  const lines = groupIntoLines([...singleLinePlate()].reverse());
  expect(lines.length).toBe(1);
  const xs = lines[0].map(d => d.x);
  expect(xs).toEqual([...xs].sort((a, b) => a - b));
});

test('empty input', () => {
  expect(groupIntoLines([])).toEqual([]);
  expect(readPlateText([], CLASS_NAMES)).toBe('');
  expect(readPlateDetailed([], CLASS_NAMES).ordered).toEqual([]);
});

test('English two line plate', () => {
  expect(readPlateText(twoLinePlate(), CLASS_NAMES)).toBe(
    'Dhaka Metro-Ga 27-8476',
  );
});

test('Bangla two line plate', () => {
  expect(readPlateText(twoLinePlate(), CLASS_NAMES, CLASS_DISPLAY)).toBe(
    'ঢাকা মেট্রো-গ ২৭-৮৪৭৬',
  );
});

test('Bangla single line plate', () => {
  expect(readPlateText(singleLinePlate(), CLASS_NAMES, CLASS_DISPLAY)).toBe(
    'ঢাকা মেট্রো-ঘ ১১-২৩৪৫',
  );
});

test('low confidence boxes are ignored below minConf', () => {
  const dets = [...twoLinePlate(), det('5', 0.9, 0.72, 0.06, 0.2, 0.1)];
  expect(readPlateText(dets, CLASS_NAMES, undefined, 0.25)).toBe(
    'Dhaka Metro-Ga 27-8476',
  );
  expect(readPlateText(dets, CLASS_NAMES, undefined, 0).endsWith('5')).toBe(
    true,
  );
});

test('partial read still gives text', () => {
  const onlyDigits = '1234'.split('').map((d, i) => det(d, 0.1 + 0.1 * i, 0.5));
  expect(readPlateText(onlyDigits, CLASS_NAMES)).toBe('1234');
});

test('unknown class id does not crash', () => {
  const dets: Detection[] = [{ cls: 999, x: 0.5, y: 0.5, w: 0.1, h: 0.2, conf: 0.9 }];
  expect(plateTokens(dets, CLASS_NAMES)).toEqual(['?']);
  expect(readPlateText(dets, CLASS_NAMES)).toBe('?');
});

test('plate tokens are in reading order', () => {
  expect(plateTokens(twoLinePlate(), CLASS_NAMES)).toEqual([
    'Dhaka',
    'Metro',
    'Ga',
    ...'278476'.split(''),
  ]);
});

test('overlapping boxes on one line keep a stable left to right order', () => {
  // Two digits overlap heavily, with the same x. Input order must be kept.
  const dets = [
    det('3', 0.5, 0.5, 0.2),
    det('4', 0.5, 0.5, 0.2),
    det('2', 0.3, 0.5, 0.2),
  ];
  expect(plateTokens(dets, CLASS_NAMES)).toEqual(['2', '3', '4']);
});

test('boxes with the same y but huge overlap still form one line', () => {
  const dets = [det('1', 0.2, 0.5), det('2', 0.21, 0.51), det('3', 0.22, 0.49)];
  expect(groupIntoLines(dets).length).toBe(1);
});

test('line split threshold is half the median box height', () => {
  // median height 0.2 gives a threshold of 0.1. A gap of exactly 0.1 stays on
  // the same line, a gap above 0.1 starts a new one.
  const same = [det('1', 0.2, 0.5), det('2', 0.4, 0.6)];
  expect(groupIntoLines(same).length).toBe(1);
  const split = [det('1', 0.2, 0.5), det('2', 0.4, 0.61)];
  expect(groupIntoLines(split).length).toBe(2);
});

test('detailed read returns ordered detections and tokens', () => {
  const r = readPlateDetailed(twoLinePlate(), CLASS_NAMES, CLASS_DISPLAY);
  expect(r.text).toBe('ঢাকা মেট্রো-গ ২৭-৮৪৭৬');
  expect(r.ordered.length).toBe(9);
  expect(r.tokens).toEqual(['Dhaka', 'Metro', 'Ga', ...'278476'.split('')]);
  expect(r.ordered.map(d => CLASS_NAMES[d.cls])).toEqual(r.tokens);
});

test('edit distance', () => {
  expect(editDistance(['a', 'b'], ['a', 'b'])).toBe(0);
  expect(editDistance(['a', 'b', 'c'], ['a', 'c'])).toBe(1);
  expect(editDistance([], ['x', 'y'])).toBe(2);
});

test('median', () => {
  expect(median([])).toBe(0);
  expect(median([3, 1, 2])).toBe(2);
  expect(median([4, 1, 3, 2])).toBe(2.5);
});

test('display map covers every class name', () => {
  expect(CLASS_NAMES.filter(n => !(n in CLASS_DISPLAY))).toEqual([]);
});

// A plate seen from a tilted phone is a level plate rotated by some angle.
const tilted = (dets: Detection[], deg: number) => rotateDetections(dets, deg);

test('estimateTilt is zero for level plates', () => {
  expect(estimateTilt(twoLinePlate())).toBe(0);
  expect(estimateTilt(singleLinePlate())).toBe(0);
  expect(estimateTilt(twoLinePlate().slice(0, 3))).toBe(0); // too few boxes
});

test('a tilted two line plate reads the same as a level one', () => {
  const level = plateTokens(twoLinePlate(), CLASS_NAMES);
  for (const deg of [-25, -15, -8, 8, 15, 25]) {
    expect(plateTokens(tilted(twoLinePlate(), deg), CLASS_NAMES)).toEqual(level);
  }
});

test('a tilted single line plate reads the same as a level one', () => {
  const level = plateTokens(singleLinePlate(), CLASS_NAMES);
  for (const deg of [-20, 12, 30]) {
    expect(plateTokens(tilted(singleLinePlate(), deg), CLASS_NAMES)).toEqual(level);
  }
});

test('estimateTilt finds the angle', () => {
  // Rotating by +15 degrees must be undone by about -15.
  expect(Math.abs(estimateTilt(tilted(twoLinePlate(), 15)) + 15)).toBeLessThanOrEqual(3);
});

test('plain grouping breaks when tilted but the reader does not', () => {
  const plate = tilted(twoLinePlate(), 20);
  expect(groupIntoLines(plate).length).toBeGreaterThan(2);
  expect(readPlateText(plate, CLASS_NAMES)).toBe(
    readPlateText(twoLinePlate(), CLASS_NAMES),
  );
});
