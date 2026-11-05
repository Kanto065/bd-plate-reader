/**
 * Pure pixel helpers for the plate reader: JPEG header parsing, EXIF
 * orientation, cropping and letterboxing into a model input tensor.
 *
 * Images are RGB, 3 bytes per pixel, row by row. No native code is involved,
 * so everything here runs under Jest.
 */

export interface RgbImage {
  data: Uint8Array | Uint8ClampedArray;
  width: number;
  height: number;
}

/** Pixel box with corners (x1, y1) top left and (x2, y2) bottom right. */
export interface PixelBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

const readU16 = (b: Uint8Array, o: number, little: boolean): number =>
  little ? b[o] | (b[o + 1] << 8) : (b[o] << 8) | b[o + 1];

const readU32 = (b: Uint8Array, o: number, little: boolean): number =>
  little
    ? (b[o] | (b[o + 1] << 8) | (b[o + 2] << 16) | (b[o + 3] << 24)) >>> 0
    : ((b[o] << 24) | (b[o + 1] << 16) | (b[o + 2] << 8) | b[o + 3]) >>> 0;

/** Walk the JPEG marker segments and call fn(marker, segmentStart, length). */
function eachSegment(
  bytes: Uint8Array,
  fn: (marker: number, start: number, length: number) => boolean | void,
): void {
  if (bytes.length < 4 || bytes[0] !== 0xff || bytes[1] !== 0xd8) {
    return;
  }
  let off = 2;
  while (off + 4 <= bytes.length) {
    if (bytes[off] !== 0xff) {
      return;
    }
    const marker = bytes[off + 1];
    if (marker === 0xff) {
      off += 1; // fill byte
      continue;
    }
    if (marker === 0xd9 || marker === 0xda) {
      return;
    }
    const length = (bytes[off + 2] << 8) | bytes[off + 3];
    if (fn(marker, off + 4, length - 2) === true) {
      return;
    }
    off += 2 + length;
  }
}

/** True when the bytes start with the JPEG signature. */
export function isJpeg(bytes: Uint8Array): boolean {
  return bytes.length > 3 && bytes[0] === 0xff && bytes[1] === 0xd8;
}

/** Image size from the JPEG header, or null if it cannot be found. */
export function readJpegSize(
  bytes: Uint8Array,
): { width: number; height: number } | null {
  let result: { width: number; height: number } | null = null;
  eachSegment(bytes, (marker, start) => {
    const isSof =
      marker >= 0xc0 &&
      marker <= 0xcf &&
      marker !== 0xc4 &&
      marker !== 0xc8 &&
      marker !== 0xcc;
    if (isSof && start + 5 <= bytes.length) {
      result = {
        height: (bytes[start + 1] << 8) | bytes[start + 2],
        width: (bytes[start + 3] << 8) | bytes[start + 4],
      };
      return true;
    }
    return false;
  });
  return result;
}

/** EXIF orientation 1 to 8 from a JPEG, or 1 when there is none. */
export function readExifOrientation(bytes: Uint8Array): number {
  let orientation = 1;
  eachSegment(bytes, (marker, start, length) => {
    if (marker !== 0xe1 || length < 14) {
      return false;
    }
    const isExif =
      bytes[start] === 0x45 &&
      bytes[start + 1] === 0x78 &&
      bytes[start + 2] === 0x69 &&
      bytes[start + 3] === 0x66 &&
      bytes[start + 4] === 0 &&
      bytes[start + 5] === 0;
    if (!isExif) {
      return false;
    }
    const tiff = start + 6;
    const little = bytes[tiff] === 0x49 && bytes[tiff + 1] === 0x49;
    const ifd = tiff + readU32(bytes, tiff + 4, little);
    if (ifd + 2 > bytes.length) {
      return true;
    }
    const count = readU16(bytes, ifd, little);
    for (let i = 0; i < count; i++) {
      const entry = ifd + 2 + i * 12;
      if (entry + 12 > bytes.length) {
        break;
      }
      if (readU16(bytes, entry, little) === 0x0112) {
        const value = readU16(bytes, entry + 8, little);
        if (value >= 1 && value <= 8) {
          orientation = value;
        }
        break;
      }
    }
    return true;
  });
  return orientation;
}

/**
 * Apply an EXIF orientation (1 to 8) so the returned image is upright.
 * Orientation 1 returns the same image object.
 */
export function applyExifOrientation(
  img: RgbImage,
  orientation: number,
): RgbImage {
  if (orientation === 1 || orientation < 1 || orientation > 8) {
    return img;
  }
  const { width: w, height: h, data } = img;
  const swap = orientation >= 5;
  const dw = swap ? h : w;
  const dh = swap ? w : h;
  const out = new Uint8Array(dw * dh * 3);
  for (let dy = 0; dy < dh; dy++) {
    for (let dx = 0; dx < dw; dx++) {
      let sx: number;
      let sy: number;
      switch (orientation) {
        case 2:
          sx = w - 1 - dx;
          sy = dy;
          break;
        case 3:
          sx = w - 1 - dx;
          sy = h - 1 - dy;
          break;
        case 4:
          sx = dx;
          sy = h - 1 - dy;
          break;
        case 5:
          sx = dy;
          sy = dx;
          break;
        case 6:
          sx = dy;
          sy = h - 1 - dx;
          break;
        case 7:
          sx = w - 1 - dy;
          sy = h - 1 - dx;
          break;
        default: // 8
          sx = w - 1 - dy;
          sy = dx;
          break;
      }
      const s = (sy * w + sx) * 3;
      const d = (dy * dw + dx) * 3;
      out[d] = data[s];
      out[d + 1] = data[s + 1];
      out[d + 2] = data[s + 2];
    }
  }
  return { data: out, width: dw, height: dh };
}

/** Copy a pixel box out of an image. The box is rounded and clamped. */
export function cropRgb(img: RgbImage, box: PixelBox): RgbImage {
  const x1 = Math.max(0, Math.min(img.width - 1, Math.floor(box.x1)));
  const y1 = Math.max(0, Math.min(img.height - 1, Math.floor(box.y1)));
  const x2 = Math.max(x1 + 1, Math.min(img.width, Math.ceil(box.x2)));
  const y2 = Math.max(y1 + 1, Math.min(img.height, Math.ceil(box.y2)));
  const width = x2 - x1;
  const height = y2 - y1;
  const out = new Uint8Array(width * height * 3);
  for (let y = 0; y < height; y++) {
    const src = ((y1 + y) * img.width + x1) * 3;
    out.set(img.data.subarray(src, src + width * 3), y * width * 3);
  }
  return { data: out, width, height };
}

export interface Letterboxed {
  /** NHWC float32 tensor values in 0..1, size x size x 3. */
  tensor: Float32Array;
  size: number;
  scale: number;
  padX: number;
  padY: number;
}

/** Gray used by Ultralytics for letterbox padding, 114 of 255. */
export const LETTERBOX_PAD = 114 / 255;

/**
 * Resize keeping the aspect ratio and pad to size x size, like Ultralytics
 * letterbox. Uses bilinear sampling. Returns a float32 NHWC tensor in 0..1.
 */
export function letterbox(img: RgbImage, size: number): Letterboxed {
  const scale = Math.min(size / img.width, size / img.height);
  const newW = Math.max(1, Math.round(img.width * scale));
  const newH = Math.max(1, Math.round(img.height * scale));
  const padX = Math.floor((size - newW) / 2);
  const padY = Math.floor((size - newH) / 2);
  const tensor = new Float32Array(size * size * 3).fill(LETTERBOX_PAD);
  const maxX = img.width - 1;
  const maxY = img.height - 1;
  const data = img.data;

  // Precompute the horizontal sample positions once per column.
  const x0s = new Int32Array(newW);
  const x1s = new Int32Array(newW);
  const fxs = new Float32Array(newW);
  for (let x = 0; x < newW; x++) {
    const sx = Math.min(maxX, Math.max(0, (x + 0.5) / scale - 0.5));
    const x0 = Math.floor(sx);
    x0s[x] = x0;
    x1s[x] = Math.min(maxX, x0 + 1);
    fxs[x] = sx - x0;
  }

  for (let y = 0; y < newH; y++) {
    const sy = Math.min(maxY, Math.max(0, (y + 0.5) / scale - 0.5));
    const y0 = Math.floor(sy);
    const y1 = Math.min(maxY, y0 + 1);
    const fy = sy - y0;
    const row0 = y0 * img.width;
    const row1 = y1 * img.width;
    let out = ((y + padY) * size + padX) * 3;
    for (let x = 0; x < newW; x++) {
      const fx = fxs[x];
      const a = (row0 + x0s[x]) * 3;
      const b = (row0 + x1s[x]) * 3;
      const c = (row1 + x0s[x]) * 3;
      const d = (row1 + x1s[x]) * 3;
      for (let ch = 0; ch < 3; ch++) {
        const top = data[a + ch] * (1 - fx) + data[b + ch] * fx;
        const bottom = data[c + ch] * (1 - fx) + data[d + ch] * fx;
        tensor[out++] = (top * (1 - fy) + bottom * fy) / 255;
      }
    }
  }
  return { tensor, size, scale, padX, padY };
}

/**
 * Convert a box given in normalized letterboxed coordinates (0..1 of the model
 * input) to pixel coordinates of the original image.
 */
export function unletterboxBox(
  lb: Pick<Letterboxed, 'size' | 'scale' | 'padX' | 'padY'>,
  cx: number,
  cy: number,
  w: number,
  h: number,
): PixelBox {
  const toX = (n: number) => (n * lb.size - lb.padX) / lb.scale;
  const toY = (n: number) => (n * lb.size - lb.padY) / lb.scale;
  return {
    x1: toX(cx - w / 2),
    y1: toY(cy - h / 2),
    x2: toX(cx + w / 2),
    y2: toY(cy + h / 2),
  };
}

/** Grow a pixel box by a fraction of its own size on each side, then clamp. */
export function padBox(
  box: PixelBox,
  fraction: number,
  width: number,
  height: number,
): PixelBox {
  const dx = (box.x2 - box.x1) * fraction;
  const dy = (box.y2 - box.y1) * fraction;
  return {
    x1: Math.max(0, box.x1 - dx),
    y1: Math.max(0, box.y1 - dy),
    x2: Math.min(width, box.x2 + dx),
    y2: Math.min(height, box.y2 + dy),
  };
}
