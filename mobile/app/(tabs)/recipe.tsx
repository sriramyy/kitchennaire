import { Image } from 'expo-image';
import { IconSymbol } from '@/components/ui/icon-symbol';
import React, { useState } from 'react';
import { ActivityIndicator, Pressable, ScrollView, StyleSheet, Text, TextInput, View } from 'react-native';
import { useRouter } from 'expo-router';
import Animated, { FadeInDown, FadeInUp } from 'react-native-reanimated';

import { useKitchen } from '@/components/KitchenContext';

const AnimatedPressable = Animated.createAnimatedComponent(Pressable);

export default function RecipeScreen() {
  const { recipe, loadRecipeFromLink, loading } = useKitchen();
  const [link, setLink] = useState('');
  const router = useRouter();

  const onLoad = async () => {
    await loadRecipeFromLink(link);
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
              onChangeText={setLink}
              placeholderTextColor="#666"
            />
          </View>
          <Pressable 
            style={[styles.btn, loading && styles.btnDisabled]} 
            onPress={onLoad}
            disabled={loading}>
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
                onPress={() => {
                  setLink(item.link);
                  loadRecipeFromLink(item.link);
                }}>
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

      {recipe && (
        <Animated.View entering={FadeInUp} style={styles.recipeCard}>
          <View style={styles.recipeHeader}>
            <Text style={styles.recipeTitle}>{recipe.title}</Text>
            <View style={styles.recipeMeta}>
              <View style={styles.metaItem}>
                <IconSymbol name="clock.fill" size={16} color="#666" />
                <Text style={styles.metaText}>20 min</Text>
              </View>
              <View style={styles.metaItem}>
                <IconSymbol name="person.2.fill" size={16} color="#666" />
                <Text style={styles.metaText}>4 servings</Text>
              </View>
            </View>
          </View>

          <View style={styles.ingredientsContainer}>
            <Text style={styles.ingredientsTitle}>
              <IconSymbol name="list.bullet" size={20} color="#000" /> Ingredients
            </Text>
            {recipe.ingredients.map((ing) => (
              <View key={ing} style={styles.ingredientRow}>
                <IconSymbol name="checkmark.circle" size={20} color="#32d74b" />
                <Text style={styles.ingredient}>{ing}</Text>
              </View>
            ))}
          </View>

          <View style={styles.buttonRow}>
            <Pressable style={styles.secondaryButton}>
              <IconSymbol name="square.and.arrow.up" size={20} color="#ff6b6b" />
              <Text style={styles.secondaryButtonText}>Share</Text>
            </Pressable>
            
            <Pressable style={styles.secondaryButton}>
              <IconSymbol name="heart" size={20} color="#ff6b6b" />
              <Text style={styles.secondaryButtonText}>Save</Text>
            </Pressable>

            <AnimatedPressable
              entering={FadeInUp.delay(300)}
              style={styles.cookButton}
              onPress={() => router.push('/cooking' as any)}
              accessibilityLabel="Start Cooking">
              <IconSymbol name="play.fill" size={20} color="#fff" />
              <Text style={styles.cookButtonText}>Start Cooking</Text>
            </AnimatedPressable>
          </View>
        </Animated.View>
      )}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { 
    flex: 1,
    backgroundColor: '#f5f5f5'
  },
  content: {
    padding: 16,
    gap: 16,
  },
  header: {
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
