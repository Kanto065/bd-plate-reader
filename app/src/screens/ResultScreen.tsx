import React, { useEffect, useRef, useState } from 'react';
import {
  ActivityIndicator,
  Image,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
} from 'react-native';
import Clipboard from '@react-native-clipboard/clipboard';
import type { NativeStackScreenProps } from '@react-navigation/native-stack';
import type { RootStackParamList } from '../navigation/types';
import { readPlate } from '../services/readPlate';
import { pickPhoto } from '../services/pickPhoto';

type Props = NativeStackScreenProps<RootStackParamList, 'Result'>;

export default function ResultScreen({ navigation, route }: Props) {
  const { imageUri, source } = route.params;
  const [text, setText] = useState('');
  const [loading, setLoading] = useState(true);
  const [copied, setCopied] = useState(false);
  const copiedTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    readPlate(imageUri)
      .then(result => {
        if (!cancelled) {
          setText(result);
        }
      })
      .catch(() => {
        if (!cancelled) {
          setText('');
        }
      })
      .finally(() => {
        if (!cancelled) {
          setLoading(false);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [imageUri]);

  useEffect(
    () => () => {
      if (copiedTimer.current) {
        clearTimeout(copiedTimer.current);
      }
    },
    [],
  );

  const onCopy = () => {
    Clipboard.setString(text);
    setCopied(true);
    if (copiedTimer.current) {
      clearTimeout(copiedTimer.current);
    }
    copiedTimer.current = setTimeout(() => setCopied(false), 1500);
  };

  const onRetake = async () => {
    if (source === 'camera') {
      navigation.replace('Camera');
      return;
    }
    const uri = await pickPhoto();
    if (uri) {
      navigation.replace('Result', { imageUri: uri, source: 'gallery' });
    }
  };

  return (
    <ScrollView contentContainerStyle={styles.container}>
      <Image
        source={{ uri: imageUri }}
        style={styles.preview}
        resizeMode="contain"
      />
      <Text style={styles.label}>Plate text (you can edit it)</Text>
      {loading ? (
        <ActivityIndicator size="large" color="#0b6e4f" style={styles.loader} />
      ) : (
        <TextInput
          style={styles.input}
          value={text}
          onChangeText={setText}
          multiline
          placeholder="No plate text found"
        />
      )}
      <Pressable
        style={[styles.primary, (loading || !text) && styles.disabled]}
        onPress={onCopy}
        disabled={loading || !text}>
        <Text style={styles.primaryText}>{copied ? 'Copied' : 'Copy'}</Text>
      </Pressable>
      <Pressable style={styles.secondary} onPress={onRetake}>
        <Text style={styles.secondaryText}>Retake</Text>
      </Pressable>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { padding: 20, backgroundColor: '#ffffff', flexGrow: 1 },
  preview: {
    width: '100%',
    height: 240,
    borderRadius: 12,
    backgroundColor: '#eeeeee',
  },
  label: { fontSize: 14, color: '#555555', marginTop: 20, marginBottom: 6 },
  loader: { marginVertical: 24 },
  input: {
    borderWidth: 1,
    borderColor: '#cccccc',
    borderRadius: 10,
    padding: 14,
    fontSize: 26,
    color: '#111111',
    minHeight: 64,
    marginBottom: 16,
  },
  primary: {
    backgroundColor: '#0b6e4f',
    paddingVertical: 14,
    borderRadius: 12,
    alignItems: 'center',
    marginBottom: 12,
  },
  disabled: { opacity: 0.5 },
  primaryText: { color: '#ffffff', fontSize: 18, fontWeight: '600' },
  secondary: {
    borderWidth: 2,
    borderColor: '#0b6e4f',
    paddingVertical: 12,
    borderRadius: 12,
    alignItems: 'center',
  },
  secondaryText: { color: '#0b6e4f', fontSize: 18, fontWeight: '600' },
});
