import React from 'react';
import { Pressable, StyleSheet, Text, View } from 'react-native';
import type { NativeStackScreenProps } from '@react-navigation/native-stack';
import type { RootStackParamList } from '../navigation/types';
import { pickPhoto } from '../services/pickPhoto';

type Props = NativeStackScreenProps<RootStackParamList, 'Home'>;

export default function HomeScreen({ navigation }: Props) {
  const onUpload = async () => {
    const uri = await pickPhoto();
    if (uri) {
      navigation.navigate('Result', { imageUri: uri, source: 'gallery' });
    }
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>BD Plate Reader</Text>
      <Text style={styles.subtitle}>
        Read a Bangladeshi number plate from a photo. Everything runs on your
        phone.
      </Text>
      <Pressable
        style={styles.primary}
        onPress={() => navigation.navigate('Camera')}>
        <Text style={styles.primaryText}>Take photo</Text>
      </Pressable>
      <Pressable style={styles.secondary} onPress={onUpload}>
        <Text style={styles.secondaryText}>Upload photo</Text>
      </Pressable>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    padding: 24,
    backgroundColor: '#ffffff',
  },
  title: { fontSize: 28, fontWeight: '700', color: '#111111' },
  subtitle: { fontSize: 16, color: '#555555', marginTop: 8, marginBottom: 32 },
  primary: {
    backgroundColor: '#0b6e4f',
    paddingVertical: 16,
    borderRadius: 12,
    alignItems: 'center',
    marginBottom: 12,
  },
  primaryText: { color: '#ffffff', fontSize: 18, fontWeight: '600' },
  secondary: {
    borderWidth: 2,
    borderColor: '#0b6e4f',
    paddingVertical: 14,
    borderRadius: 12,
    alignItems: 'center',
  },
  secondaryText: { color: '#0b6e4f', fontSize: 18, fontWeight: '600' },
});
