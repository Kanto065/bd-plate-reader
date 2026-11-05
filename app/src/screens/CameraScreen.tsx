import React, { useEffect, useRef, useState } from 'react';
import {
  ActivityIndicator,
  Linking,
  Pressable,
  StyleSheet,
  Text,
  View,
} from 'react-native';
import {
  Camera,
  useCameraDevice,
  useCameraFormat,
  useCameraPermission,
} from 'react-native-vision-camera';
import type { NativeStackScreenProps } from '@react-navigation/native-stack';
import type { RootStackParamList } from '../navigation/types';
import { MAX_SIDE } from '../ml/imageLoader';

type Props = NativeStackScreenProps<RootStackParamList, 'Camera'>;

export default function CameraScreen({ navigation }: Props) {
  const { hasPermission, requestPermission } = useCameraPermission();
  const device = useCameraDevice('back');
  // Ask for a photo close to 1600 x 1200. The plate reader decodes the JPEG in
  // JavaScript, so a full resolution photo (up to 48 MP) would be far too slow.
  const format = useCameraFormat(device, [
    { photoResolution: { width: MAX_SIDE, height: (MAX_SIDE * 3) / 4 } },
  ]);
  const camera = useRef<Camera>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!hasPermission) {
      requestPermission();
    }
  }, [hasPermission, requestPermission]);

  const capture = async () => {
    if (!camera.current || busy) {
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const photo = await camera.current.takePhoto({ flash: 'off' });
      navigation.replace('Result', {
        imageUri: `file://${photo.path}`,
        source: 'camera',
      });
    } catch (e) {
      console.warn('takePhoto failed', e);
      setError(e instanceof Error ? e.message : 'Could not take the photo.');
      setBusy(false);
    }
  };

  if (!hasPermission) {
    return (
      <View style={styles.center}>
        <Text style={styles.message}>Camera permission is needed.</Text>
        <Pressable style={styles.button} onPress={requestPermission}>
          <Text style={styles.buttonText}>Allow camera</Text>
        </Pressable>
        <Pressable style={styles.button} onPress={() => Linking.openSettings()}>
          <Text style={styles.buttonText}>Open settings</Text>
        </Pressable>
      </View>
    );
  }

  if (!device) {
    return (
      <View style={styles.center}>
        <Text style={styles.message}>No camera found on this device.</Text>
      </View>
    );
  }

  return (
    <View style={styles.fill}>
      <Camera
        ref={camera}
        style={StyleSheet.absoluteFill}
        device={device}
        format={format}
        isActive
        photo
      />
      {error ? <Text style={styles.errorText}>{error}</Text> : null}
      <View style={styles.bar}>
        {busy ? (
          <ActivityIndicator color="#ffffff" size="large" />
        ) : (
          <Pressable style={styles.shutter} onPress={capture} />
        )}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  fill: { flex: 1, backgroundColor: '#000000' },
  center: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    padding: 24,
    backgroundColor: '#ffffff',
  },
  message: { fontSize: 16, color: '#333333', marginBottom: 16 },
  button: {
    backgroundColor: '#0b6e4f',
    paddingVertical: 12,
    paddingHorizontal: 24,
    borderRadius: 10,
    marginTop: 8,
  },
  buttonText: { color: '#ffffff', fontSize: 16, fontWeight: '600' },
  errorText: {
    position: 'absolute',
    top: 60,
    left: 16,
    right: 16,
    color: '#ffffff',
    backgroundColor: '#b00020',
    padding: 10,
    borderRadius: 8,
  },
  bar: {
    position: 'absolute',
    bottom: 40,
    left: 0,
    right: 0,
    alignItems: 'center',
  },
  shutter: {
    width: 76,
    height: 76,
    borderRadius: 38,
    backgroundColor: '#ffffff',
    borderWidth: 6,
    borderColor: '#0b6e4f',
  },
});
