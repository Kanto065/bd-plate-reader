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
import { loadRgbImage } from '../ml/imageLoader';
import { loadModels } from '../ml/modelLoader';
import { runPipeline } from '../ml/plateReader';

export const PLACEHOLDER_PLATE = 'ঢাকা মেট্রো-ঘ ১১-২৩৪৫';

export interface PlateResult {
  text: string;
  lowConfidence: boolean;
  /** Set when reading failed, so the screen can say why instead of staying blank. */
  error?: string;
}

export async function readPlate(imageUri: string): Promise<PlateResult> {
  let models;
  try {
    models = await loadModels();
  } catch (error) {
    console.warn('Could not load the models', error);
    return { text: '', lowConfidence: true, error: 'The plate models failed to load.' };
  }
  if (!models) {
    await new Promise<void>(resolve => setTimeout(resolve, 400));
    return { text: PLACEHOLDER_PLATE, lowConfidence: false };
  }
  try {
    const image = await loadRgbImage(imageUri);
    const result = await runPipeline(models, image);
    return { text: result.text, lowConfidence: result.lowConfidence };
  } catch (error) {
    console.warn('Plate reading failed', error);
    return { text: '', lowConfidence: true, error: 'This photo could not be read.' };
  }
}
