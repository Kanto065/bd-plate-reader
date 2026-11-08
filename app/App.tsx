import React, { useEffect } from 'react';
import { Linking, StatusBar } from 'react-native';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { SafeAreaProvider } from 'react-native-safe-area-context';
import type { RootStackParamList } from './src/navigation/types';
import HomeScreen from './src/screens/HomeScreen';
import CameraScreen from './src/screens/CameraScreen';
import ResultScreen from './src/screens/ResultScreen';
import { loadModels } from './src/ml/modelLoader';
import { SELFTEST_URL, runSelfTest } from './src/dev/selfTest';

const Stack = createNativeStackNavigator<RootStackParamList>();

function App() {
  // Load and warm up the models now so the first photo is not slow. A failure
  // is reported later, when a photo is read.
  useEffect(() => {
    loadModels().catch(() => {});
  }, []);

  // Test only: bdplate://selftest reads the photos in the app's selftest folder.
  useEffect(() => {
    const handle = (url?: string | null) => {
      if (url && url.startsWith(SELFTEST_URL)) {
        runSelfTest();
      }
    };
    Promise.resolve(Linking.getInitialURL()).then(handle);
    const sub = Linking.addEventListener('url', e => handle(e.url));
    return () => sub?.remove?.();
  }, []);

  return (
    <SafeAreaProvider>
      <StatusBar barStyle="dark-content" />
      <NavigationContainer>
        <Stack.Navigator initialRouteName="Home">
          <Stack.Screen
            name="Home"
            component={HomeScreen}
            options={{ title: 'BD Plate Reader' }}
          />
          <Stack.Screen
            name="Camera"
            component={CameraScreen}
            options={{ title: 'Take photo' }}
          />
          <Stack.Screen
            name="Result"
            component={ResultScreen}
            options={{ title: 'Result' }}
          />
        </Stack.Navigator>
      </NavigationContainer>
    </SafeAreaProvider>
  );
}

export default App;
