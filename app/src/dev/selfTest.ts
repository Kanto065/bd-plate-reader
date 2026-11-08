/**
 * Test only entry point. Runs the plate reader on every photo in the app's own
 * external folder Android/data/<package>/files/selftest and logs one SELFTEST
 * line per photo, so a phone can be tested from a PC without the photo picker.
 *
 *   adb shell mkdir -p /sdcard/Android/data/com.bdplatereader/files/selftest
 *   adb push photo.jpg /sdcard/Android/data/com.bdplatereader/files/selftest/
 *   adb shell am start -a android.intent.action.VIEW -d bdplate://selftest
 *   adb logcat -s ReactNativeJS | grep SELFTEST
 *
 * The photos stay on the phone, the output goes to logcat only.
 */
import { NativeModules } from 'react-native';
import { readPlate } from '../services/readPlate';

export const SELFTEST_URL = 'bdplate://selftest';

let running = false;

export async function runSelfTest(): Promise<void> {
  const native = NativeModules.PlateImage;
  if (running || !native?.testFiles) {
    return;
  }
  running = true;
  try {
    const files: string[] = await native.testFiles();
    for (const file of files) {
      const started = Date.now();
      const r = await readPlate(file);
      console.log(
        'SELFTEST',
        JSON.stringify({
          file: file.split('/').pop(),
          tokens: (r.tokens ?? []).join(' '),
          text: r.text,
          low: r.lowConfidence,
          reasons: r.reasons ?? [],
          error: r.error ?? null,
          ms: Date.now() - started,
        }),
      );
    }
    console.log('SELFTEST_DONE', files.length);
  } finally {
    running = false;
  }
}
