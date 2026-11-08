jest.mock('../src/ml/modelLoader', () => ({
  loadModels: jest.fn(),
}));
jest.mock('../src/ml/imageLoader', () => ({
  loadRgbImage: jest.fn(),
}));

import { readPlate } from '../src/services/readPlate';
import { loadModels } from '../src/ml/modelLoader';

test('readPlate reports an error when the models fail to load', async () => {
  jest.spyOn(console, 'warn').mockImplementation(() => {});
  (loadModels as jest.Mock).mockRejectedValue(new Error('no protocol'));
  const result = await readPlate('file://x.jpg');
  expect(result.text).toBe('');
  expect(result.lowConfidence).toBe(true);
  expect(result.error).toBe('The plate models failed to load.');
});
