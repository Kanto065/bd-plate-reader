import { readPlate, PLACEHOLDER_PLATE } from '../src/services/readPlate';

test('readPlate returns the placeholder plate when no models are bundled', async () => {
  await expect(readPlate('file://x.jpg')).resolves.toEqual({
    text: PLACEHOLDER_PLATE,
    lowConfidence: false,
  });
});
