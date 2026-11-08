import { launchImageLibrary } from 'react-native-image-picker';

/**
 * Opens the gallery and returns the chosen image URI, or null if cancelled.
 * The photo is not resized here. The native decoder subsamples it while
 * decoding, which is cheaper than a resize and re-encode in the picker.
 */
export async function pickPhoto(): Promise<string | null> {
  const result = await launchImageLibrary({
    mediaType: 'photo',
    selectionLimit: 1,
  });
  if (result.didCancel || result.errorCode) {
    return null;
  }
  return result.assets?.[0]?.uri ?? null;
}
