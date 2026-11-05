import { launchImageLibrary } from 'react-native-image-picker';

/** Opens the gallery and returns the chosen image URI, or null if cancelled. */
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
