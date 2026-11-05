/**
 * Reads the plate text from an image.
 *
 * Milestone 1 placeholder: returns a fixed string after a short delay so the
 * UI flow can be tested end to end. Milestone 3 replaces the body with the
 * on-device pipeline (resize, plate detector, crop, character detector,
 * post-processing). The signature stays the same.
 */
export const PLACEHOLDER_PLATE = 'ঢাকা মেট্রো-ঘ ১১-২৩৪৫';

export async function readPlate(_imageUri: string): Promise<string> {
  await new Promise<void>(resolve => setTimeout(resolve, 400));
  return PLACEHOLDER_PLATE;
}
