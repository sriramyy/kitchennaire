import { Image } from 'expo-image';
import { StyleSheet } from 'react-native';

import ParallaxScrollView from '@/components/parallax-scroll-view';
import { ThemedText } from '@/components/themed-text';
import { ThemedView } from '@/components/themed-view';

export default function HomeScreen() {
  return (
    <ParallaxScrollView
      headerBackgroundColor={{ light: '#FFE1C6', dark: '#2C1810' }}
      headerImage={
        <Image
          source={{ uri: 'https://images.unsplash.com/photo-1556909114-f6e7ad7d3136?q=80&w=2670&auto=format&fit=crop' }}
          style={styles.kitchenImage}
          contentFit="cover"
        />
      }>
      <ThemedView style={styles.titleContainer}>
        <ThemedText type="title">Welcome to Kitchennaire!</ThemedText>
      </ThemedView>

      <ThemedView style={styles.sectionContainer}>
        <ThemedText type="subtitle">Your Smart Kitchen Assistant</ThemedText>
        <ThemedText>
          Kitchennaire helps you manage recipes, track ingredients, and cook with hands-free controls.
          Start your culinary journey with these easy steps:
        </ThemedText>
      </ThemedView>

      <ThemedView style={styles.featureContainer}>
        <ThemedText type="subtitle">🎯 Recipe Management</ThemedText>
        <ThemedText>
          • Paste YouTube recipe links{'\n'}
          • Get instant ingredient lists{'\n'}
          • Use hands-free cooking mode
        </ThemedText>
      </ThemedView>

      <ThemedView style={styles.featureContainer}>
        <ThemedText type="subtitle">📦 Smart Pantry</ThemedText>
        <ThemedText>
          • Scan your ingredients{'\n'}
          • Keep track of what you have{'\n'}
          • Quick-add common items
        </ThemedText>
      </ThemedView>

      <ThemedView style={styles.featureContainer}>
        <ThemedText type="subtitle">🛒 Shopping Made Easy</ThemedText>
        <ThemedText>
          • Auto-generated shopping lists{'\n'}
          • Mark items as you buy them{'\n'}
          • Clear view of what you need
        </ThemedText>
      </ThemedView>

      <ThemedView style={styles.getStartedContainer}>
        <ThemedText type="subtitle">Get Started!</ThemedText>
        <ThemedText>
          Head to the Recipe tab and paste a YouTube cooking video link to begin your cooking adventure.
        </ThemedText>
      </ThemedView>
    </ParallaxScrollView>
  );
}

const styles = StyleSheet.create({
  titleContainer: {
    alignItems: 'center',
    marginBottom: 16,
  },
  sectionContainer: {
    gap: 8,
    marginBottom: 24,
    backgroundColor: 'rgba(255,225,198,0.1)',
    padding: 16,
    borderRadius: 12,
  },
  featureContainer: {
    gap: 8,
    marginBottom: 16,
  },
  getStartedContainer: {
    gap: 8,
    marginTop: 8,
    backgroundColor: 'rgba(161,206,220,0.1)',
    padding: 16,
    borderRadius: 12,
  },
  kitchenImage: {
    height: '100%',
    width: '100%',
    position: 'absolute',
  },
});
