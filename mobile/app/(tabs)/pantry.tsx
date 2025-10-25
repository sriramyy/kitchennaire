import React, { useState } from 'react';
import { ActivityIndicator, Pressable, ScrollView, StyleSheet, Text, TextInput, View } from 'react-native';
import { IconSymbol } from '@/components/ui/icon-symbol';
import { useKitchen } from '@/components/KitchenContext';
import Animated, { FadeInDown, FadeInUp, Layout } from 'react-native-reanimated';

const AnimatedPressable = Animated.createAnimatedComponent(Pressable);

const QUICK_ADD_ITEMS = [
  'Salt', 'Pepper', 'Olive Oil', 'Flour', 'Sugar', 
  'Eggs', 'Milk', 'Butter', 'Garlic', 'Onion',
  'Tomato', 'Potato', 'Rice', 'Pasta'
];

const CATEGORIES = [
  { name: 'All', icon: 'square.grid.2x2' },
  { name: 'Staples', icon: 'cart' },
  { name: 'Produce', icon: 'leaf' },
  { name: 'Dairy', icon: 'drop' },
  { name: 'Spices', icon: 'flame' },
];

export default function PantryScreen() {
  const { pantry, scanPantry, uploadPhoto, loading, addPantryItem } = useKitchen();
  const [selectedCategory, setSelectedCategory] = useState('All');
  const [searchQuery, setSearchQuery] = useState('');

  const filteredPantry = pantry.filter(item => 
    item.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      <Animated.View entering={FadeInDown.delay(100)} style={styles.header}>
        <Text style={styles.headerTitle}>My Pantry</Text>
        <Text style={styles.headerSubtitle}>Keep track of your ingredients</Text>
      </Animated.View>

      <Animated.View entering={FadeInDown.delay(200)} style={styles.searchContainer}>
        <IconSymbol name="magnifyingglass" size={20} color="#666" />
        <TextInput
          placeholder="Search ingredients..."
          style={styles.searchInput}
          value={searchQuery}
          onChangeText={setSearchQuery}
          placeholderTextColor="#666"
        />
      </Animated.View>

      <ScrollView 
        horizontal 
        showsHorizontalScrollIndicator={false} 
        style={styles.categoriesScroll}
        contentContainerStyle={styles.categoriesContent}>
        {CATEGORIES.map(category => (
          <Pressable
            key={category.name}
            style={[
              styles.categoryTab,
              selectedCategory === category.name && styles.categoryTabActive
            ]}
            onPress={() => setSelectedCategory(category.name)}>
            <IconSymbol 
              name={category.icon} 
              size={20} 
              color={selectedCategory === category.name ? '#fff' : '#666'} 
            />
            <Text style={[
              styles.categoryText,
              selectedCategory === category.name && styles.categoryTextActive
            ]}>
              {category.name}
            </Text>
          </Pressable>
        ))}
      </ScrollView>

      <View style={styles.actionsContainer}>
        <AnimatedPressable 
          entering={FadeInUp.delay(300)}
          style={styles.mainAction}
          onPress={() => scanPantry()}>
          <IconSymbol name="camera.fill" size={24} color="#fff" />
          <Text style={styles.mainActionText}>Scan Ingredients</Text>
        </AnimatedPressable>

        <AnimatedPressable 
          entering={FadeInUp.delay(400)}
          style={styles.secondaryAction}
          onPress={() => uploadPhoto()}>
          <IconSymbol name="photo.fill" size={24} color="#ff6b6b" />
          <Text style={styles.secondaryActionText}>Upload Photo</Text>
        </AnimatedPressable>
      </View>

      {loading && (
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color="#ff6b6b" />
          <Text style={styles.loadingText}>Scanning ingredients...</Text>
        </View>
      )}

      <Animated.View entering={FadeInUp.delay(500)} style={styles.pantryContainer}>
        <View style={styles.sectionHeader}>
          <Text style={styles.sectionTitle}>Current Ingredients</Text>
          <Text style={styles.count}>{filteredPantry.length} items</Text>
        </View>

        {filteredPantry.length === 0 ? (
          <Text style={styles.emptyText}>
            {searchQuery ? 'No matching ingredients found.' : 'No items yet. Scan or add ingredients to get started.'}
          </Text>
        ) : (
          <View style={styles.itemsGrid}>
            {filteredPantry.map((item) => (
              <Animated.View
                key={item}
                entering={FadeInUp}
                layout={Layout}
                style={styles.itemCard}>
                <IconSymbol name="checkmark.circle.fill" size={20} color="#32d74b" />
                <Text style={styles.itemText}>{item}</Text>
              </Animated.View>
            ))}
          </View>
        )}
      </Animated.View>

      <View style={styles.quickAddSection}>
        <Text style={styles.sectionTitle}>Quick Add</Text>
        <View style={styles.quickAddGrid}>
          {QUICK_ADD_ITEMS.map((item) => (
            <Pressable
              key={item}
              style={[
                styles.quickAddBtn,
                pantry.includes(item) && styles.quickAddBtnDisabled
              ]}
              onPress={() => addPantryItem(item)}
              disabled={pantry.includes(item)}>
              <Text style={[
                styles.quickAddText,
                pantry.includes(item) && styles.quickAddTextDisabled
              ]}>
                {pantry.includes(item) ? '✓' : '+'} {item}
              </Text>
            </Pressable>
          ))}
        </View>
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
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
  searchContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 12,
    gap: 8,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 2,
  },
  searchInput: {
    flex: 1,
    fontSize: 16,
  },
  categoriesScroll: {
    marginHorizontal: -16,
  },
  categoriesContent: {
    paddingHorizontal: 16,
    gap: 8,
  },
  categoryTab: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    paddingVertical: 8,
    paddingHorizontal: 16,
    borderRadius: 20,
    backgroundColor: '#fff',
  },
  categoryTabActive: {
    backgroundColor: '#ff6b6b',
  },
  categoryText: {
    fontSize: 15,
    color: '#666',
    fontWeight: '500',
  },
  categoryTextActive: {
    color: '#fff',
  },
  actionsContainer: {
    flexDirection: 'row',
    gap: 12,
  },
  mainAction: {
    flex: 2,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    backgroundColor: '#ff6b6b',
    padding: 16,
    borderRadius: 12,
  },
  mainActionText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  secondaryAction: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    backgroundColor: '#fff',
    padding: 16,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: '#ff6b6b',
  },
  secondaryActionText: {
    color: '#ff6b6b',
    fontSize: 16,
    fontWeight: '600',
  },
  loadingContainer: {
    padding: 20,
    alignItems: 'center',
  },
  loadingText: {
    marginTop: 8,
    color: '#666',
  },
  pantryContainer: {
    backgroundColor: '#fff',
    borderRadius: 16,
    padding: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 3,
  },
  sectionHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: '600',
  },
  count: {
    fontSize: 14,
    color: '#666',
  },
  emptyText: {
    color: '#666',
    textAlign: 'center',
    marginVertical: 20,
  },
  itemsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  itemCard: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    backgroundColor: '#f5f5f5',
    paddingVertical: 8,
    paddingHorizontal: 12,
    borderRadius: 8,
  },
  itemText: {
    fontSize: 15,
  },
  quickAddSection: {
    marginTop: 8,
  },
  quickAddGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
    marginTop: 12,
  },
  quickAddBtn: {
    backgroundColor: '#fff',
    paddingVertical: 8,
    paddingHorizontal: 12,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#ff6b6b',
  },
  quickAddBtnDisabled: {
    backgroundColor: '#f5f5f5',
    borderColor: '#ccc',
  },
  quickAddText: {
    color: '#ff6b6b',
    fontWeight: '500',
  },
  quickAddTextDisabled: {
    color: '#666',
  },
});
