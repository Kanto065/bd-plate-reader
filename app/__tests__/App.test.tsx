// The real models are bundled in the app, so pretend they are not for this test.
jest.mock('../src/ml/modelAssets', () => ({ PLATE_MODEL: null, CHAR_MODEL: null }));

import { readPlate, PLACEHOLDER_PLATE } from '../src/services/readPlate';

test('readPlate returns the placeholder plate when no models are bundled', async () => {
  await expect(readPlate('file://x.jpg')).resolves.toEqual({
    text: PLACEHOLDER_PLATE,
    lowConfidence: false,
  });
});
