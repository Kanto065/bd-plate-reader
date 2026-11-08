/**
 * Reads the plate text from an image.
 *
 * Returns the text and a low confidence flag. The Result screen shows
 * "Check the characters before copying" when the flag is set.
 *
 * When the trained models are bundled, this runs the on-device pipeline. When
 * they are not (the models are copied in with `npm run link:models`), it
 * returns a fixed placeholder after a short delay so the UI flow still works.
 * If the pipeline fails on a real photo, the result is empty text flagged as
 * low confidence. The placeholder is never shown in that case.
 */
import { loadImageSource } from '../ml/imageLoader';
import { loadModels } from '../ml/modelLoader';
import { runPipeline } from '../ml/plateReader';
import { getTiming, resetTiming, timedAsync } from '../ml/timing';

export const PLACEHOLDER_PLATE = 'ঢাকা মেট্রো-ঘ ১১-২৩৪৫';

export interface PlateResult {
  text: string;
  lowConfidence: boolean;
  /** Set when reading failed, so the screen can say why instead of staying blank. */
  error?: string;
  /** Why the read is low confidence, in plain words. */
  reasons?: string[];
  /** Class names in reading order, used by the self test. */
  tokens?: string[];
}

export async function readPlate(imageUri: string): Promise<PlateResult> {
  resetTiming();
  const startedAt = Date.now();
  let models;
  try {
    models = await timedAsync('modelLoad', loadModels);
  } catch (error) {
    console.warn('Could not load the models', error);
    return { text: '', lowConfidence: true, error: 'The plate models failed to load.' };
  }
  if (!models) {
    await new Promise<void>(resolve => setTimeout(resolve, 400));
    return { text: PLACEHOLDER_PLATE, lowConfidence: false };
  }
  try {
    const image = await loadImageSource(imageUri);
    const result = await runPipeline(models, image);
    console.log(
      'PLATE_TIMING',
      JSON.stringify({ totalMs: Date.now() - startedAt, ...getTiming() }),
    );
    return {
      text: result.text,
      lowConfidence: result.lowConfidence,
      reasons: result.reasons,
      tokens: result.tokens,
    };
  } catch (error) {
    console.warn('Plate reading failed', error);
    return { text: '', lowConfidence: true, error: 'This photo could not be read.' };
  }
}
