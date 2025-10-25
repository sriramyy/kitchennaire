import { View, StyleSheet, Dimensions } from 'react-native';
import { useKitchen } from '@/components/KitchenContext';
import { Camera, CameraType } from 'expo-camera';
import WebView from 'react-native-webview';
import { useEffect, useState } from 'react';
import { ThemedView } from '@/components/themed-view';
import { ThemedText } from '@/components/themed-text';
import { IconSymbol } from '@/components/ui/icon-symbol';

const { height } = Dimensions.get('window');

export default function CookingMode() {
  const { recipe } = useKitchen();
  const [hasPermission, setHasPermission] = useState<boolean | null>(null);

  useEffect(() => {
    (async () => {
      const { status } = await Camera.requestCameraPermissionsAsync();
      setHasPermission(status === 'granted');
    })();
  }, []);

  if (!recipe) {
    return (
      <ThemedView style={styles.container}>
        <IconSymbol name="questionmark.circle" size={48} color="#999" />
        <ThemedText style={styles.message}>No recipe selected</ThemedText>
      </ThemedView>
    );
  }

  if (hasPermission === null) {
    return (
      <ThemedView style={styles.container}>
        <ThemedText>Requesting camera permission...</ThemedText>
      </ThemedView>
    );
  }

  if (hasPermission === false) {
    return (
      <ThemedView style={styles.container}>
        <IconSymbol name="exclamationmark.circle" size={48} color="#ff3b30" />
        <ThemedText style={styles.message}>Camera access denied</ThemedText>
      </ThemedView>
    );
  }

  // Extract video ID from YouTube URL
  const videoId = recipe.youtubeUrl?.match(/(?:youtu\.be\/|youtube\.com\/(?:embed\/|v\/|watch\?v=|watch\?.+&v=))([^&?/]+)/)?.[1];

  return (
    <View style={styles.container}>
      {/* YouTube Embed */}
      <View style={styles.videoContainer}>
        {videoId ? (
          <WebView
            style={styles.webview}
            javaScriptEnabled={true}
            domStorageEnabled={true}
            source={{
              uri: `https://www.youtube.com/embed/${videoId}?playsinline=1&modestbranding=1&controls=1`,
            }}
          />
        ) : (
          <ThemedView style={[styles.webview, styles.noVideo]}>
            <IconSymbol name="play.slash" size={48} color="#999" />
            <ThemedText style={styles.message}>No video available</ThemedText>
          </ThemedView>
        )}
      </View>

      {/* Camera View */}
      <View style={styles.cameraContainer}>
        <Camera
          style={styles.camera}
          type={CameraType.front}
        />
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  videoContainer: {
    height: height / 2,
    backgroundColor: '#000',
  },
  webview: {
    flex: 1,
  },
  noVideo: {
    alignItems: 'center',
    justifyContent: 'center',
    gap: 12,
  },
  cameraContainer: {
    flex: 1,
    backgroundColor: '#000',
  },
  camera: {
    flex: 1,
  },
  message: {
    fontSize: 16,
    marginTop: 8,
    textAlign: 'center',
  },

});
