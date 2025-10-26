import { Image } from 'expo-image';
import { IconSymbol } from '@/components/ui/icon-symbol';
import React, { useState } from 'react';
import { ActivityIndicator, Pressable, ScrollView, StyleSheet, Text, TextInput, View, Platform } from 'react-native';
import { useRouter } from 'expo-router';
import Animated, { FadeInDown, FadeInUp } from 'react-native-reanimated';
import { useKitchen } from '@/components/KitchenContext'; // Imports the updated Recipe type

const AnimatedPressable = Animated.createAnimatedComponent(Pressable);

// Prefer putting this in an env var: EXPO_PUBLIC_API_BASE_URL
// For iOS simulator: http://127.0.0.1:8000
// For Android emulator: http://10.0.2.2:8000
// For physical device: http://<your-computer-LAN-IP>:8000
// Using hotspot IP
const API_BASE = 'http://172.20.10.2:8000'; // please dont leak
// TODO: put in env

export default function RecipeScreen() {
  const { recipe, loadRecipeFromLink, loading, setCurrentVideoId } = useKitchen();
  const [link, setLink] = useState('');
  const [error, setError] = useState('');
  const router = useRouter();

  const sendUrl = async (url: string) => {
    try {
      const payload = { yt_url: url };
      const jsonBody = JSON.stringify(payload);
      
      console.log('Sending request to:', `${API_BASE}/submit_url`);
      console.log('Payload object:', payload);
      console.log('JSON string being sent:', jsonBody);
      
      const res = await fetch(`${API_BASE}/submit_url`, {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        },
        body: jsonBody,
      });

      console.log('Response status:', res.status);
      const text = await res.text();
      console.log('Raw response:', text);

      const data = JSON.parse(text);
      console.log('Backend responded:', data);
      
      if (data.video_id) {
        setCurrentVideoId(data.video_id);
        console.log('Updated video ID:', data.video_id);
      }
      
      return data;
    } catch (error: any) {
      console.error('Failed to submit URL:', error);
      if (error?.name) console.error('Error type:', error.name);
      if (error?.message) console.error('Error message:', error.message);
      if (error instanceof TypeError) {
        console.error('Network error - check if backend is reachable at:', API_BASE);
      }
      throw error;
    }
  };

  const onLoad = async () => {
    setError(''); // Clear any previous errors
    const url = link.trim();
    if (!url) {
      setError('Please enter a YouTube URL');
      return;
    }
    console.log('Starting onLoad with URL:', url);
    try {
      // send to backend (will print on FastAPI console)
      const result = await sendUrl(url);
      console.log('sendUrl completed with result:', result);
      // keep your existing behavior 
      // populates 'recipe' with the structured data.
      await loadRecipeFromLink(url);
    } catch (error: any) {
      console.error('onLoad error:', error);
      // Show the backend's error message or a fallback
      const backendError = error?.message?.includes('detail') 
        ? JSON.parse(error.message).detail
        : 'Failed to submit URL. Please enter a valid YouTube URL';
      setError(backendError);
    }
  };

  const quickRecipes = [
    { name: 'Quick Pasta', link: 'https://youtu.be/example1' },
    { name: '15min Chicken', link: 'https://youtu.be/example2' },
    { name: 'Easy Salad', link: 'https://youtu.be/example3' },
  ];

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      <Animated.View entering={FadeInDown.delay(100)} style={styles.header}>
        <Text style={styles.headerTitle}>Find a Recipe</Text>
        <Text style={styles.headerSubtitle}>Paste a YouTube link or choose from quick recipes</Text>
      </Animated.View>

      <Animated.View entering={FadeInDown.delay(200)} style={styles.card}>
        <View style={styles.inputRow}>
          <View style={styles.inputContainer}>
            <IconSymbol name="link" size={20} color="#666" />
            <TextInput
              placeholder="Paste YouTube recipe link"
              style={styles.input}
              value={link}
              onChangeText={(text) => {
                setLink(text);
                setError(''); // Clear error when user types
              }}
              placeholderTextColor="#666"
              autoCapitalize="none"
              autoCorrect={false}
            />
          </View>
          {error ? <Text style={styles.errorText}>{error}</Text> : null}
          <Pressable
            style={[styles.btn, loading && styles.btnDisabled]}
            onPress={onLoad}
            disabled={loading}
          >
            <Text style={styles.btnText}>{loading ? 'Loading...' : 'Get Recipe'}</Text>
          </Pressable>
        </View>

        <View style={styles.quickRecipes}>
          <Text style={styles.quickTitle}>Quick Recipes</Text>
          <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.quickScroll}>
            {quickRecipes.map((item, index) => (
              <Pressable
                key={item.name}
                style={styles.quickCard}
                onPress={async () => {
                  setLink(item.link);
                  await sendUrl(item.link);
                  await loadRecipeFromLink(item.link);
                }}
              >
                <Image
                  source={{ uri: `https://picsum.photos/200/200?random=${index}` }}
                  style={styles.quickImage}
                />
                <Text style={styles.quickText}>{item.name}</Text>
              </Pressable>
            ))}
          </ScrollView>
        </View>
      </Animated.View>

      {loading && (
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color="#ff6b6b" />
          <Text style={styles.loadingText}>Loading recipe...</Text>
        </View>
      )}

      {/* The Recipe Card: Uses data imported via the useKitchen context (which 
        is populated by your backend API call).
      */}
      {recipe && (
        <Animated.View entering={FadeInUp} style={styles.recipeCard}>
          <View style={styles.recipeHeader}>
            <Text style={styles.recipeTitle}>{recipe.title}</Text>
            
            <View style={styles.recipeMeta}>
              {/* Time (using recipe.timeInMinutes) */}
              <View style={styles.metaItem}>
                <IconSymbol name="timer" size={16} color="#666" />
                <Text style={styles.metaText}>{recipe.timeInMinutes} min</Text>
              </View>
              
              {/* Video Duration (using recipe.videoDuration) */}
              <View style={styles.metaItem}>
                <IconSymbol name="video" size={16} color="#666" />
                <Text style={styles.metaText}>{recipe.videoDuration}</Text>
              </View>
            </View>
          </View>

          {/* Ingredients List */}
          <View style={styles.ingredientsContainer}>
            <Text style={styles.ingredientsTitle}>Ingredients</Text>
            {/* Map over the ingredients array from the recipe object */}
            {recipe.ingredients.map((item, index) => (
              <View key={index} style={styles.ingredientRow}>
                <IconSymbol 
                  name={item.isAvailable ? 'checkmark.circle.fill' : 'xmark.circle.fill'} 
                  size={20} 
                  color={item.isAvailable ? '#ff6b6b' : '#666'} 
                />
                <Text style={styles.ingredient}>{item.name}</Text>
              </View>
            ))}
          </View>

          {/* Action Buttons */}
          <View style={styles.buttonRow}>
            {/* Save Recipe Button */}
            <Pressable style={styles.secondaryButton}>
              <IconSymbol name="bookmark" size={20} color="#ff6b6b" />
              <Text style={styles.secondaryButtonText}>Save</Text>
            </Pressable>
            
            {/* Start Recipe Button */}
            <Pressable 
              style={styles.cookButton} 
              onPress={() => router.push('/cook-mode')} 
            >
              <IconSymbol name="play" size={20} color="white" />
              <Text style={styles.cookButtonText}>Start Cooking</Text>
            </Pressable>
          </View>
        </Animated.View>
      )}
    </ScrollView>
  );
}
const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
  },
  errorText: {
    color: '#ff3b30',
    fontSize: 14,
    marginTop: 4,
    marginLeft: 8,
  },
  content: {
    padding: 16,
    gap: 16,
  },
  header: {
    marginTop: 55,
    marginBottom: 8,
  },
  headerTitle: {
    fontSize: 28,
    fontWeight: '700',
    color: '#000',
  },
  headerSubtitle: {
    fontSize: 16,
    color: '#666',
    marginTop: 4,
  },
  card: {
    backgroundColor: '#fff',
    borderRadius: 16,
    padding: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 3,
  },
  inputContainer: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#f5f5f5',
    borderRadius: 12,
    padding: 12,
    gap: 8,
  },
  inputRow: { 
    flexDirection: 'row',
    gap: 12,
  },
  input: { 
    flex: 1,
    fontSize: 16,
  },
  btn: { 
    backgroundColor: '#ff6b6b',
    paddingHorizontal: 20,
    justifyContent: 'center',
    borderRadius: 12,
  },
  btnDisabled: {
    backgroundColor: '#ffb1b1',
  },
  btnText: { 
    color: 'white',
    fontWeight: '600',
    fontSize: 16,
  },
  quickRecipes: {
    marginTop: 16,
  },
  quickTitle: {
    fontSize: 18,
    fontWeight: '600',
    marginBottom: 12,
  },
  quickScroll: {
    marginHorizontal: -16,
    paddingHorizontal: 16,
  },
  quickCard: {
    marginRight: 12,
    width: 120,
    backgroundColor: '#fff',
    borderRadius: 12,
    overflow: 'hidden',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 2,
  },
  quickImage: {
    width: '100%',
    height: 80,
  },
  quickText: {
    padding: 8,
    fontSize: 14,
    fontWeight: '500',
  },
  loadingContainer: {
    padding: 20,
    alignItems: 'center',
  },
  loadingText: {
    marginTop: 8,
    color: '#666',
  },
  recipeCard: {
    backgroundColor: '#fff',
    borderRadius: 16,
    padding: 20,
    gap: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 3,
  },
  recipeHeader: {
    gap: 12,
  },
  recipeTitle: {
    fontSize: 24,
    fontWeight: '700',
  },
  recipeMeta: {
    flexDirection: 'row',
    gap: 16,
  },
  metaItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
  },
  metaText: {
    color: '#666',
    fontSize: 14,
  },
  ingredientsContainer: {
    gap: 12,
  },
  ingredientsTitle: {
    fontSize: 18,
    fontWeight: '600',
    marginBottom: 4,
  },
  ingredientRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    paddingVertical: 4,
  },
  ingredient: {
    fontSize: 16,
  },
  buttonRow: {
    flexDirection: 'row',
    gap: 12,
    marginTop: 8,
  },
  secondaryButton: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 6,
    padding: 12,
    borderRadius: 12,
    backgroundColor: '#fff',
    borderWidth: 1,
    borderColor: '#ff6b6b',
  },
  secondaryButtonText: {
    color: '#ff6b6b',
    fontWeight: '600',
    fontSize: 15,
  },
  cookButton: {
    flex: 2,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 6,
    padding: 12,
    borderRadius: 12,
    backgroundColor: '#ff6b6b',
  },
  cookButtonText: {
    color: 'white',
    fontWeight: '700',
    fontSize: 16,
  },
});
