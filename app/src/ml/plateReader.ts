/**
 * The two stage plate reading pipeline.
 *
 *   image -> letterbox -> plate detector -> best box -> padded crop
 *         -> letterbox -> character detector -> NMS -> post-processing text
 *
 * The models are reached through the small ModelHandle interface, so this file
 * has no native imports and is covered by Jest with fake models. The adapter
 * for react-native-fast-tflite is in modelLoader.ts.
 */
import { assessConfidence } from './confidence';
import { CLASS_DISPLAY, CLASS_NAMES } from './classes';
import {
  ImageSource,
  Letterboxed,
  RgbImage,
  hwcToChw,
  padBox,
  rgbSource,
  unletterboxBox,
} from './imageOps';
import { readPlateDetailed } from './postprocess';
import { QuantParams, bufferToFloat, floatToBuffer } from './quant';
import { timed, timedAsync } from './timing';
import { decodeYolo, nms } from './yolo';

export interface TensorInfo {
  dataType: string;
  shape: number[];
}

/** What the pipeline needs from a loaded TFLite model. */
export interface ModelHandle {
  inputs: TensorInfo[];
  outputs: TensorInfo[];
  run(input: ArrayBuffer[]): Promise<ArrayBuffer[]>;
}

export interface StageConfig {
  /** Square model input side in pixels. Must match the exported model. */
  inputSize: number;
  scoreThreshold: number;
  iouThreshold: number;
  maxDetections: number;
  /** Suppress overlapping boxes of different classes too. */
  agnosticNms: boolean;
  /** Only needed when the model input is int8 or uint8. */
  inputQuant?: QuantParams;
  /** Needed when the model output is int8 or uint8. */
  outputQuant?: QuantParams;
}

export interface PipelineConfig {
  plate: StageConfig;
  chars: StageConfig;
  /** Extra margin around the plate box, as a fraction of its size per side. */
  platePadding: number;
  /** Character boxes below this score are dropped before reading. */
  charMinConf: number;
  /** A plate box scoring below this flags the read as low confidence. */
  plateLowScore: number;
}

export const DEFAULT_PIPELINE_CONFIG: PipelineConfig = {
  plate: {
    inputSize: 640,
    scoreThreshold: 0.25,
    iouThreshold: 0.5,
    maxDetections: 5,
    agnosticNms: true,
  },
  chars: {
    inputSize: 416,
    scoreThreshold: 0.25,
    iouThreshold: 0.5,
    maxDetections: 20,
    agnosticNms: true,
  },
  platePadding: 0.05,
  charMinConf: 0.25,
  plateLowScore: 0.3,
};

export interface ModelPair {
  plate: ModelHandle;
  chars: ModelHandle;
}

export interface PipelineResult {
  text: string;
  lowConfidence: boolean;
  /** False when the plate detector found nothing. */
  plateFound: boolean;
  reasons: string[];
}

async function detect(
  model: ModelHandle,
  lb: Letterboxed,
  stage: StageConfig,
  name: string,
) {
  const inType = model.inputs[0]?.dataType ?? 'float32';
  // The LiteRT exports take planar input [1, 3, H, W]. Older TFLite exports
  // take [1, H, W, 3]. The letterbox tensor is HWC, so reorder when needed.
  const inShape = model.inputs[0]?.shape ?? [];
  const planar = inShape.length === 4 && inShape[1] === 3 && inShape[3] !== 3;
  if (lb.planar && !planar) {
    throw new Error('The image is planar but the model expects NHWC');
  }
  const input = timed(name + 'Prep', () => {
    const ordered =
      planar && !lb.planar ? hwcToChw(lb.tensor, stage.inputSize) : lb.tensor;
    return floatToBuffer(ordered, inType, stage.inputQuant);
  });
  const outputs = await timedAsync(name + 'Run', () => model.run([input]));
  const info = model.outputs[0];
  if (!info || outputs.length === 0) {
    throw new Error('Model returned no output');
  }
  return timed(name + 'Decode', () => {
    const values = bufferToFloat(outputs[0], info.dataType, stage.outputQuant);
    const decoded = decodeYolo(values, info.shape, {
      inputSize: stage.inputSize,
      scoreThreshold: stage.scoreThreshold,
    });
    return nms(decoded, {
      iouThreshold: stage.iouThreshold,
      agnostic: stage.agnosticNms,
      maxDetections: stage.maxDetections,
    });
  });
}

export async function runPipeline(
  models: ModelPair,
  image: RgbImage | ImageSource,
  config: PipelineConfig = DEFAULT_PIPELINE_CONFIG,
): Promise<PipelineResult> {
  const source = 'letterbox' in image ? image : rgbSource(image);
  // Stage 1: find the plate.
  const lb = await timedAsync('plateLetterbox', () =>
    source.letterbox(config.plate.inputSize),
  );
  const plates = await detect(models.plate, lb, config.plate, 'plate');
  if (plates.length === 0) {
    return {
      text: '',
      lowConfidence: true,
      plateFound: false,
      reasons: ['no plate found'],
    };
  }
  const best = plates[0]; // nms returns the highest score first
  const box = padBox(
    unletterboxBox(lb, best.x, best.y, best.w, best.h),
    config.platePadding,
    lb.srcWidth,
    lb.srcHeight,
  );
  const crop = source.crop(box);

  // Stage 2: read the characters on the crop.
  const clb = await timedAsync('charLetterbox', () =>
    crop.letterbox(config.chars.inputSize),
  );
  const chars = await detect(models.chars, clb, config.chars, 'char');

  const read = readPlateDetailed(
    chars,
    CLASS_NAMES,
    CLASS_DISPLAY,
    config.charMinConf,
  );
  const scores = read.ordered.map(d => d.conf);
  // The plate box score counts too: an unsure plate makes the read unsure.
  const assessment = assessConfidence(read.tokens, scores);
  const reasons = [...assessment.reasons];
  let low = assessment.low;
  if (best.conf < config.plateLowScore) {
    low = true;
    reasons.push('plate detection score is low');
  }
  return { text: read.text, lowConfidence: low, plateFound: true, reasons };
}
