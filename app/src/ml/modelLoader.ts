/**
 * Loads the two TFLite models with react-native-fast-tflite and adapts them to
 * the ModelHandle interface used by the pipeline.
 *
 * react-native-fast-tflite is required lazily inside loadModels so that
 * importing this file never touches native code. That keeps Jest and the
 * placeholder path free of native dependencies.
 */
import { CHAR_MODEL, PLATE_MODEL } from './modelAssets';
import type { ModelHandle, ModelPair } from './plateReader';

let cached: Promise<ModelPair | null> | null = null;

/** True when both model files are bundled with the app. */
export function modelsBundled(): boolean {
  return PLATE_MODEL !== null && CHAR_MODEL !== null;
}

async function load(source: number): Promise<ModelHandle> {
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  const { loadTensorflowModel } = require('react-native-fast-tflite');
  // An empty delegate list means the default CPU delegate. Try 'nnapi' or
  // 'android-gpu' later once accuracy and speed are measured on the device.
  const model = await loadTensorflowModel(source, []);
  return {
    inputs: model.inputs,
    outputs: model.outputs,
    run: (input: ArrayBuffer[]) => model.run(input),
  };
}

/** Resolves to null when the models are not bundled. Loads once, then caches. */
export function loadModels(): Promise<ModelPair | null> {
  if (!modelsBundled()) {
    return Promise.resolve(null);
  }
  if (!cached) {
    cached = Promise.all([
      load(PLATE_MODEL as number),
      load(CHAR_MODEL as number),
    ])
      .then(([plate, chars]) => ({ plate, chars }))
      .catch(error => {
        cached = null; // allow a retry on the next photo
        throw error;
      });
  }
  return cached;
}
