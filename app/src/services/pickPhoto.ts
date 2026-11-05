import { launchImageLibrary } from 'react-native-image-picker';
import { MAX_SIDE } from '../ml/imageLoader';

/**
 * Opens the gallery and returns the chosen image URI, or null if cancelled.
 * The picker downsizes the photo, because the plate reader decodes the JPEG
 * in JavaScript and needs a small image.
 */
export async function pickPhoto(): Promise<string | null> {
  const result = await launchImageLibrary({
    mediaType: 'photo',
    selectionLimit: 1,
    maxWidth: MAX_SIDE,
    maxHeight: MAX_SIDE,
    quality: 0.9,
  });
  if (result.didCancel || result.errorCode) {
    return null;
  }
  return result.assets?.[0]?.uri ?? null;
}
