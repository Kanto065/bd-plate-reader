/**
 * Quantization helpers for int8 and uint8 TFLite tensors.
 *
 * Quantized value q relates to the real value r by r = scale * (q - zeroPoint).
 * react-native-fast-tflite reports each tensor's data type and shape but not
 * its scale and zero point, so those come from configuration. How to read them
 * from the exported model is described in docs/ml_integration.md.
 */

export interface QuantParams {
  scale: number;
  zeroPoint: number;
}

export type QuantType = 'int8' | 'uint8';

export function quantize(
  values: Float32Array,
  q: QuantParams,
  type: QuantType,
): Int8Array | Uint8Array {
  const lo = type === 'int8' ? -128 : 0;
  const hi = type === 'int8' ? 127 : 255;
  const out = type === 'int8' ? new Int8Array(values.length) : new Uint8Array(values.length);
  for (let i = 0; i < values.length; i++) {
    const v = Math.round(values[i] / q.scale + q.zeroPoint);
    out[i] = v < lo ? lo : v > hi ? hi : v;
  }
  return out;
}

export function dequantize(
  values: Int8Array | Uint8Array,
  q: QuantParams,
): Float32Array {
  const out = new Float32Array(values.length);
  for (let i = 0; i < values.length; i++) {
    out[i] = q.scale * (values[i] - q.zeroPoint);
  }
  return out;
}

/** Tensor data types the reader can handle. */
export type TensorType = 'float32' | 'int8' | 'uint8';

/** Turn a raw output buffer into floats, dequantizing integer types. */
export function bufferToFloat(
  buffer: ArrayBuffer,
  type: string,
  q?: QuantParams,
): Float32Array {
  if (type === 'float32') {
    return new Float32Array(buffer);
  }
  if (type === 'int8' || type === 'uint8') {
    if (!q) {
      throw new Error(
        'Output tensor is ' + type + ' but no quantization parameters are configured',
      );
    }
    return dequantize(
      type === 'int8' ? new Int8Array(buffer) : new Uint8Array(buffer),
      q,
    );
  }
  throw new Error('Unsupported tensor type: ' + type);
}

/** Turn float input values into the buffer the model expects. */
export function floatToBuffer(
  values: Float32Array,
  type: string,
  q?: QuantParams,
): ArrayBuffer {
  if (type === 'float32') {
    return values.buffer.slice(
      values.byteOffset,
      values.byteOffset + values.byteLength,
    ) as ArrayBuffer;
  }
  if (type === 'int8' || type === 'uint8') {
    const params = q ?? { scale: 1 / 255, zeroPoint: type === 'int8' ? -128 : 0 };
    return quantize(values, params, type).buffer as ArrayBuffer;
  }
  throw new Error('Unsupported tensor type: ' + type);
}
