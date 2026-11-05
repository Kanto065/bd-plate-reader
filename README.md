# bd-plate-reader

On-device Bangladeshi number plate reader (React Native, TFLite). See CLAUDE.md for the full plan.

## Run the app on an Android phone

One time setup:
1. Install JDK 17 (Temurin) and the Android SDK. Set `JAVA_HOME` to the JDK 17 folder and `ANDROID_HOME` to the SDK folder. Add `%ANDROID_HOME%\platform-tools` to `PATH`.
2. On the phone, enable Developer options and USB debugging, connect it by USB, and accept the debugging prompt.
3. Check `adb devices` lists the phone as `device`.

Each run, from the `app/` folder:
```
npm install
npx react-native start
```
In a second terminal, also in `app/`:
```
npx react-native run-android
```
Notes:
- The Android build is limited to arm64-v8a (`reactNativeArchitectures` in `app/android/gradle.properties`). Add `x86_64` there to use an emulator.
- `app/android/local.properties` is machine specific and not committed. Create it with `sdk.dir=<path to your Android SDK>` if Gradle cannot find the SDK.
- Milestone 1 returns a placeholder plate string. Real recognition arrives in Milestone 3.
- iOS strings are configured but iOS is not built or tested.
