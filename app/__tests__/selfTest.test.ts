jest.mock('../src/services/readPlate', () => ({
  readPlate: jest.fn(),
}));

import { NativeModules } from 'react-native';
import { readPlate } from '../src/services/readPlate';
import { runSelfTest } from '../src/dev/selfTest';
import { assessConfidence } from '../src/ml/confidence';

afterEach(() => {
  delete NativeModules.PlateImage;
  jest.restoreAllMocks();
});

test('self test reads every file and logs one SELFTEST line each', async () => {
  const log = jest.spyOn(console, 'log').mockImplementation(() => {});
  NativeModules.PlateImage = {
    testFiles: jest.fn().mockResolvedValue(['file:///d/a.jpg', 'file:///d/b.jpg']),
  };
  (readPlate as jest.Mock)
    .mockResolvedValueOnce({ text: 'ঢাকা', lowConfidence: false, tokens: ['Dhaka', 'Metro'], reasons: [] })
    .mockResolvedValueOnce({ text: '', lowConfidence: true, error: 'This photo could not be read.' });
  await runSelfTest();
  const lines = log.mock.calls.filter(c => c[0] === 'SELFTEST').map(c => JSON.parse(c[1]));
  expect(lines.map(l => l.file)).toEqual(['a.jpg', 'b.jpg']);
  expect(lines[0].tokens).toBe('Dhaka Metro');
  expect(lines[1].error).toBe('This photo could not be read.');
  expect(log).toHaveBeenCalledWith('SELFTEST_DONE', 2);
});

test('self test does nothing without the native module', async () => {
  const log = jest.spyOn(console, 'log').mockImplementation(() => {});
  await runSelfTest();
  expect(log).not.toHaveBeenCalled();
});

test('the low score reason says which score and the limit', () => {
  const r = assessConfidence(
    ['Dhaka', 'Metro', 'Ga', '1', '2', '3', '4', '5', '6'],
    [0.9, 0.9, 0.53, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9],
  );
  expect(r.low).toBe(true);
  expect(r.reasons).toEqual(['lowest character score 0.53 is under 0.6']);
});
