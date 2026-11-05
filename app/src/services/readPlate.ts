/**
 * Reads the plate text from an image.
 *
 * Returns the text and a low confidence flag. The Result screen shows
 * "Check the characters before copying" when the flag is set.
 *
 * Until the trained models are bundled, this returns a fixed placeholder after
 * a short delay so the UI flow can be tested end to end.
 */
export const PLACEHOLDER_PLATE = 'ঢাকা মেট্রো-ঘ ১১-২৩৪৫';

export interface PlateResult {
  text: string;
  lowConfidence: boolean;
}

export async function readPlate(_imageUri: string): Promise<PlateResult> {
  await new Promise<void>(resolve => setTimeout(resolve, 400));
  return { text: PLACEHOLDER_PLATE, lowConfidence: false };
}
