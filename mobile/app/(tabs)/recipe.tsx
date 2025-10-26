// recipe.tsx (Updated)

import { Image } from 'expo-image';
import { IconSymbol } from '@/components/ui/icon-symbol';
import React, { useState } from 'react';
import { ActivityIndicator, Pressable, ScrollView, StyleSheet, Text, TextInput, View, Platform } from 'react-native';
import { useRouter } from 'expo-router';
import Animated, { FadeInDown, FadeInUp } from 'react-native-reanimated';
import { useKitchen } from '@/components/KitchenContext'; // Imports the updated Recipe type

const AnimatedPressable = Animated.createAnimatedComponent(Pressable);

// This is no longer used directly in this file, but is fine to keep as a reference
const API_BASE = process.env.EXPO_PUBLIC_API_BASE ; // please dont leak

export default function RecipeScreen() {
  // Get the correct functions and state from the context
  const { recipe, loadRecipeFromLink, loading } = useKitchen();
  const [link, setLink] = useState('');
  const [error, setError] = useState('');
  const router = useRouter();

  const onLoad = async () => { // new onload
    setError(''); // Clear any previous errors
    const url = link.trim();
    if (!url) {
      setError('Please enter a YouTube URL');
      return;
    }
    console.log('Starting onLoad with URL:', url);
    try {
      // This one function now does everything:
      // 1. Calls /get_recipe
      // 2. Gets the full recipe with 'isAvailable'
      // 3. Sets the 'recipe' state in the context
      await loadRecipeFromLink(url);
      console.log('loadRecipeFromLink completed successfully.');

    } catch (error: any) {
      console.error('onLoad error:', error);
      // Set the error message from the hook
      setError(error.message || 'An unknown error occurred.');
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
        </View>
        {/* Error text will now appear below the input row */}
        {error ? <Text style={styles.errorText}>{error}</Text> : null}
        <Pressable
            style={[styles.btn, loading && styles.btnDisabled]}
            onPress={onLoad} // This now calls the correct onLoad
            disabled={loading}
          >
            <Text style={styles.btnText}>{loading ? 'Loading...' : 'Get Recipe'}</Text>
          </Pressable>

        <View style={styles.quickRecipes}>
          <Text style={styles.quickTitle}>Quick Recipes</Text>
          <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.quickScroll}>
            {quickRecipes.map((item, index) => (
              <Pressable
                key={item.name}
                style={styles.quickCard}
                onPress={async () => {
                  setLink(item.link); // Update text input
                  setError('');       // Clear error
                  try {
                    // Directly call the context function
                    await loadRecipeFromLink(item.link);
                  } catch (error: any) {
                    console.error('Quick recipe error:', error);
                    setError(error.message || 'Failed to load recipe.');
                  }
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

      {/* The Recipe Card: This will now appear when 'recipe' is set */}
      {recipe && (
        <Animated.View entering={FadeInUp} style={styles.recipeCard}>
          <View style={styles.recipeHeader}>
            <Text style={styles.recipeTitle}>{recipe.title}</Text>

            <View style={styles.recipeMeta}>
              {/* Time (using recipe.timeInMinutes) */}
              <View style={styles.metaItem}>
                <IconSymbol name="timer" size={16} color="#666" />
                <Text style={styles.metaText}>{recipe.timeInMinutes || 12} min</Text>
              </View>

              {/* Video Duration (using recipe.videoDuration) */}
              <View style={styles.metaItem}>
                <IconSymbol name="video" size={16} color="#666" />
                <Text style={styles.metaText}>{recipe.videoDuration || '4:22'}</Text>
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
    marginTop: 8, // Added margin
    marginBottom: 4, // Added margin
    textAlign: 'center', // Centered text
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
    gap: 12, // Added gap for spacing
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
    paddingVertical: 14, // Made button taller
    paddingHorizontal: 20,
    justifyContent: 'center',
    alignItems: 'center', // Center text
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