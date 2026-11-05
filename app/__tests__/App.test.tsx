import { readPlate, PLACEHOLDER_PLATE } from '../src/services/readPlate';

test('readPlate returns the placeholder plate', async () => {
  await expect(readPlate('file://x.jpg')).resolves.toBe(PLACEHOLDER_PLATE);
});
