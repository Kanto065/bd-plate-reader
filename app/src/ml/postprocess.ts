/**
 * Turns detected plate characters into plate text.
 *
 * Pure TypeScript port of training/postprocess.py. Keep the behavior identical:
 * the Python tests and the Jest tests use the same plate examples.
 *
 * Boxes use normalized center format, the same as YOLO labels.
 *
 * Steps:
 *   1. groupIntoLines: cluster boxes into text lines by their y center.
 *   2. each line is sorted left to right by x.
 *   3. class ids are mapped to names, then to text.
 *
 * Example (English class names):
 *   Dhaka Metro Ga      line 1: district, optional Metro, series letter
 *   2 7 8 4 7 6         line 2: digits
 * becomes "Dhaka Metro-Ga 27-8476", or with the Bangla display map
 * "ঢাকা মেট্রো-গ ২৭-৮৪৭৬".
 */

export interface Detection {
  cls: number;
  x: number; // center x, normalized 0..1
  y: number; // center y, normalized 0..1
  w: number;
  h: number;
  conf: number;
}

export type DisplayMap = Readonly<Record<string, string>>;

// Series letters in the dataset class list. Digits are 0 to 9, 'Metro' is the
// metro marker, and everything else is a district or metro city word.
export const LETTERS: ReadonlySet<string> = new Set([
  'A', 'Bha', 'Cha', 'Chha', 'DA', 'Da', 'E', 'Ga', 'Gha', 'Ha', 'Ja',
  'Jha', 'Ka', 'Kha', 'La', 'Ma', 'Na', 'Pa', 'Sa', 'Sha', 'THA', 'Ta',
  'Tha', 'U', 'Dha', 'Ba',
]);
export const METRO = 'Metro';

export function median(values: readonly number[]): number {
  const s = [...values].sort((a, b) => a - b);
  const n = s.length;
  if (n === 0) {
    return 0;
  }
  const mid = Math.floor(n / 2);
  return n % 2 === 1 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
}

// Stable sort by a numeric key. Array.prototype.sort is not guaranteed stable
// on every engine, and Python's sort is, so break ties by original index.
function stableSortBy<T>(items: readonly T[], key: (t: T) => number): T[] {
  return items
    .map((item, index) => ({ item, index, k: key(item) }))
    .sort((a, b) => (a.k - b.k) || (a.index - b.index))
    .map(e => e.item);
}

/**
 * Group detections into lines by y, each line sorted left to right.
 *
 * A new line starts when a box center is further than gapRatio times the
 * median box height away from the mean y of the current line. Lines are
 * returned top to bottom.
 */
export function groupIntoLines(
  dets: readonly Detection[],
  gapRatio: number = 0.5,
): Detection[][] {
  if (dets.length === 0) {
    return [];
  }
  const threshold = gapRatio * median(dets.map(d => d.h));
  const ordered = stableSortBy(dets, d => d.y);
  const lines: Detection[][] = [[ordered[0]]];
  for (let i = 1; i < ordered.length; i++) {
    const d = ordered[i];
    const current = lines[lines.length - 1];
    const meanY = current.reduce((sum, c) => sum + c.y, 0) / current.length;
    if (Math.abs(d.y - meanY) > threshold) {
      lines.push([d]);
    } else {
      current.push(d);
    }
  }
  return lines.map(line => stableSortBy(line, d => d.x));
}

/** Map each detection to its class name. Unknown ids become '?'. */
export function linesToNames(
  lines: readonly (readonly Detection[])[],
  names: readonly string[],
): string[][] {
  return lines.map(line =>
    line.map(d =>
      Number.isInteger(d.cls) && d.cls >= 0 && d.cls < names.length
        ? names[d.cls]
        : '?',
    ),
  );
}

export function flatten<T>(lines: readonly (readonly T[])[]): T[] {
  return lines.reduce<T[]>((acc, line) => acc.concat(line), []);
}

const DIGITS = /^[0-9]+$/;

/**
 * Build the plate string from line by line class names.
 *
 * Layout: "<district> <Metro>-<letter> <dd>-<dddd>". Pieces that are not
 * detected are skipped, so a partial read still gives readable text.
 * display maps a class name to text (for example Bangla). Without it the class
 * names themselves are used.
 */
export function formatPlate(
  linesNames: readonly (readonly string[])[],
  display?: DisplayMap,
): string {
  const show = (name: string): string =>
    display && Object.prototype.hasOwnProperty.call(display, name)
      ? display[name]
      : name;

  const words: string[] = [];
  const letters: string[] = [];
  const digits: string[] = [];
  for (const name of flatten(linesNames)) {
    if (DIGITS.test(name)) {
      digits.push(name);
    } else if (LETTERS.has(name)) {
      letters.push(name);
    } else {
      words.push(name); // district, Metro, or anything unknown
    }
  }

  let head = words.map(show).join(' ');
  if (letters.length > 0) {
    const letterText = letters.map(show).join('');
    head = head ? head + '-' + letterText : letterText;
  }

  const shown = digits.map(show);
  let number: string;
  if (shown.length > 4) {
    // Hyphen before the last four digits: 278476 becomes 27-8476.
    number =
      shown.slice(0, shown.length - 4).join('') +
      '-' +
      shown.slice(shown.length - 4).join('');
  } else {
    number = shown.join('');
  }
  return [head, number].filter(p => p.length > 0).join(' ');
}

export function readPlateText(
  dets: readonly Detection[],
  names: readonly string[],
  display?: DisplayMap,
  minConf: number = 0,
): string {
  const kept = dets.filter(d => d.conf >= minConf);
  const lines = groupIntoLines(kept);
  return formatPlate(linesToNames(lines, names), display);
}

/** Reading order token list, used for exact match scoring. */
export function plateTokens(
  dets: readonly Detection[],
  names: readonly string[],
  minConf: number = 0,
): string[] {
  const kept = dets.filter(d => d.conf >= minConf);
  return flatten(linesToNames(groupIntoLines(kept), names));
}

export function editDistance(a: readonly string[], b: readonly string[]): number {
  let prev: number[] = [];
  for (let j = 0; j <= b.length; j++) {
    prev.push(j);
  }
  for (let i = 1; i <= a.length; i++) {
    const cur: number[] = [i];
    for (let j = 1; j <= b.length; j++) {
      cur.push(
        Math.min(
          prev[j] + 1,
          cur[j - 1] + 1,
          prev[j - 1] + (a[i - 1] !== b[j - 1] ? 1 : 0),
        ),
      );
    }
    prev = cur;
  }
  return prev[prev.length - 1];
}

export interface PlateRead {
  text: string;
  /** Kept detections in reading order (line by line, left to right). */
  ordered: Detection[];
  /** Class name for each entry of ordered. */
  tokens: string[];
}

/**
 * Same as readPlateText, but also returns the detections in reading order so
 * the caller can look at per-character confidence.
 */
export function readPlateDetailed(
  dets: readonly Detection[],
  names: readonly string[],
  display?: DisplayMap,
  minConf: number = 0,
): PlateRead {
  const kept = dets.filter(d => d.conf >= minConf);
  const lines = groupIntoLines(kept);
  const namesByLine = linesToNames(lines, names);
  return {
    text: formatPlate(namesByLine, display),
    ordered: flatten(lines),
    tokens: flatten(namesByLine),
  };
}
