/**
 * Decoding of YOLO detection output tensors, plus non maximum suppression.
 *
 * Ultralytics detection models (YOLO11) produce one prediction per anchor:
 * 4 box values (center x, center y, width, height) followed by one score per
 * class. TFLite exports usually shape the output as [1, 4 + classes, anchors],
 * for example [1, 5, 8400] for one class at 640 and [1, 106, 3549] for 102
 * classes at 416. Some converters transpose it to [1, anchors, 4 + classes].
 * Both layouts are handled. This still needs checking against the real
 * exported models, see docs/ml_integration.md.
 *
 * All outputs here use the same box format as postprocess.ts: normalized
 * center format relative to the model input.
 */
import type { Detection } from './postprocess';

export type Layout = 'auto' | 'channelsFirst' | 'channelsLast';
export type CoordMode = 'auto' | 'normalized' | 'pixels';

export interface DecodeOptions {
  /** Model input side in pixels, used to normalize pixel coordinates. */
  inputSize: number;
  /** Drop predictions whose best class score is below this. */
  scoreThreshold: number;
  /** Expected class count. Checked against the tensor shape when given. */
  numClasses?: number;
  layout?: Layout;
  coords?: CoordMode;
}

export function decodeYolo(
  output: ArrayLike<number>,
  shape: readonly number[],
  opts: DecodeOptions,
): Detection[] {
  // Drop leading batch dimensions of size 1.
  const dims = shape.filter((d, i) => !(d === 1 && i < shape.length - 2));
  if (dims.length !== 2) {
    throw new Error('Unexpected output shape: [' + shape.join(', ') + ']');
  }
  let layout = opts.layout ?? 'auto';
  if (layout === 'auto') {
    layout = dims[0] < dims[1] ? 'channelsFirst' : 'channelsLast';
  }
  const channels = layout === 'channelsFirst' ? dims[0] : dims[1];
  const anchors = layout === 'channelsFirst' ? dims[1] : dims[0];
  const numClasses = channels - 4;
  if (numClasses < 1) {
    throw new Error('Output has no class channels: [' + shape.join(', ') + ']');
  }
  if (opts.numClasses !== undefined && opts.numClasses !== numClasses) {
    throw new Error(
      'Expected ' + opts.numClasses + ' classes but the output has ' + numClasses,
    );
  }

  const at = (anchor: number, channel: number): number =>
    layout === 'channelsFirst'
      ? output[channel * anchors + anchor]
      : output[anchor * channels + channel];

  let coords = opts.coords ?? 'auto';
  if (coords === 'auto') {
    // Normalized boxes never exceed about 1, pixel boxes are in the hundreds.
    let maxValue = 0;
    for (let a = 0; a < anchors; a++) {
      for (let c = 0; c < 4; c++) {
        maxValue = Math.max(maxValue, at(a, c));
      }
    }
    coords = maxValue <= 2 ? 'normalized' : 'pixels';
  }
  const norm = coords === 'pixels' ? 1 / opts.inputSize : 1;

  const result: Detection[] = [];
  for (let a = 0; a < anchors; a++) {
    let best = -1;
    let bestScore = -Infinity;
    for (let c = 0; c < numClasses; c++) {
      const s = at(a, 4 + c);
      if (s > bestScore) {
        bestScore = s;
        best = c;
      }
    }
    if (bestScore >= opts.scoreThreshold) {
      result.push({
        cls: best,
        x: at(a, 0) * norm,
        y: at(a, 1) * norm,
        w: at(a, 2) * norm,
        h: at(a, 3) * norm,
        conf: bestScore,
      });
    }
  }
  return result;
}

/** Intersection over union of two center format boxes. */
export function iou(a: Detection, b: Detection): number {
  const ax1 = a.x - a.w / 2;
  const ay1 = a.y - a.h / 2;
  const ax2 = a.x + a.w / 2;
  const ay2 = a.y + a.h / 2;
  const bx1 = b.x - b.w / 2;
  const by1 = b.y - b.h / 2;
  const bx2 = b.x + b.w / 2;
  const by2 = b.y + b.h / 2;
  const iw = Math.min(ax2, bx2) - Math.max(ax1, bx1);
  const ih = Math.min(ay2, by2) - Math.max(ay1, by1);
  if (iw <= 0 || ih <= 0) {
    return 0;
  }
  const inter = iw * ih;
  const union = a.w * a.h + b.w * b.h - inter;
  return union > 0 ? inter / union : 0;
}

export interface NmsOptions {
  iouThreshold: number;
  /** When true, boxes of different classes also suppress each other. */
  agnostic?: boolean;
  /** Keep at most this many boxes. */
  maxDetections?: number;
}

/** Greedy non maximum suppression, highest score first. */
export function nms(dets: readonly Detection[], opts: NmsOptions): Detection[] {
  const sorted = dets
    .map((d, i) => ({ d, i }))
    .sort((p, q) => q.d.conf - p.d.conf || p.i - q.i)
    .map(e => e.d);
  const kept: Detection[] = [];
  for (const d of sorted) {
    let suppressed = false;
    for (const k of kept) {
      if ((opts.agnostic || k.cls === d.cls) && iou(k, d) > opts.iouThreshold) {
        suppressed = true;
        break;
      }
    }
    if (!suppressed) {
      kept.push(d);
      if (opts.maxDetections !== undefined && kept.length >= opts.maxDetections) {
        break;
      }
    }
  }
  return kept;
}
