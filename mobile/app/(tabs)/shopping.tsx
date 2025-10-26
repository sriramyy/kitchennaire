import React from 'react';
import { Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';
import { IconSymbol } from '@/components/ui/icon-symbol';
import { useKitchen } from '@/components/KitchenContext';
import Animated, { FadeInDown, FadeInUp, Layout } from 'react-native-reanimated';

const AnimatedPressable = Animated.createAnimatedComponent(Pressable);

export default function ShoppingScreen() {
  const { recipe, pantry, boughtItems, markItemBought, clearBoughtItems } = useKitchen();

  const missing = recipe?.ingredients.filter((i) => !pantry.includes(i)) ?? [];
  const bought = missing.filter((i) => boughtItems.includes(i));
  const remaining = missing.filter((i) => !boughtItems.includes(i));

  const progress = missing.length > 0 ? (bought.length / missing.length) * 100 : 0;

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      <Animated.View entering={FadeInDown.delay(100)} style={styles.header}>
        <View>
          <Text style={styles.headerTitle}>Shopping List</Text>
          <Text style={styles.headerSubtitle}>
            {recipe ? recipe.title : 'No recipe selected'}
          </Text>
        </View>
        {missing.length > 0 && (
          <View style={styles.statsCard}>
            <View style={styles.progressBar}>
              <View style={[styles.progressFill, { width: `${progress}%` }]} />
            </View>
            <Text style={styles.progressText}>
              {bought.length} of {missing.length} items
            </Text>
          </View>
        )}
      </Animated.View>

      {missing.length === 0 ? (
        <Animated.View 
          entering={FadeInUp.delay(200)} 
          style={styles.emptyStateCard}>
          <IconSymbol name="checkmark.circle.fill" size={40} color="#32d74b" />
          <Text style={styles.emptyStateTitle}>All Set!</Text>
          <Text style={styles.emptyStateText}>
            Your pantry has everything needed for this recipe.
          </Text>
        </Animated.View>
      ) : (
        <Animated.View 
          entering={FadeInUp.delay(200)}
          style={styles.listContainer}>
          <Text style={styles.sectionTitle}>Remaining Items</Text>
          {remaining.map((item, index) => (
            <AnimatedPressable
              key={item}
              entering={FadeInUp.delay(300 + index * 50)}
              layout={Layout}
              style={styles.itemCard}
              onPress={() => markItemBought(item)}>
              <View style={styles.checkbox}>
                <IconSymbol name="circle" size={24} color="#ff6b6b" />
              </View>
              <Text style={styles.itemText}>{item}</Text>
              <Text style={styles.actionHint}>Tap when bought</Text>
            </AnimatedPressable>
          ))}

          {bought.length > 0 && (
            <View style={styles.boughtSection}>
              <View style={styles.boughtHeader}>
                <Text style={styles.sectionTitle}>Purchased</Text>
                <Pressable 
                  style={styles.clearButton}
                  onPress={clearBoughtItems}>
                  <IconSymbol name="arrow.counterclockwise" size={20} color="#ff6b6b" />
                  <Text style={styles.clearButtonText}>Reset</Text>
                </Pressable>
              </View>
              {bought.map((item, index) => (
                <AnimatedPressable
                  key={item}
                  entering={FadeInUp.delay(300 + index * 50)}
                  layout={Layout}
                  style={[styles.itemCard, styles.itemCardBought]}
                  onPress={() => markItemBought(item)}>
                  <View style={styles.checkbox}>
                    <IconSymbol name="checkmark.circle.fill" size={24} color="#32d74b" />
                  </View>
                  <Text style={[styles.itemText, styles.itemTextBought]}>{item}</Text>
                </AnimatedPressable>
              ))}
            </View>
          )}
        </Animated.View>
      )}

      {missing.length > 0 && (
        <View style={styles.actionsContainer}>
          <AnimatedPressable 
            entering={FadeInUp.delay(400)}
            style={styles.shareButton}
            onPress={() => alert('Share list (to be implemented)')}>
            <IconSymbol name="square.and.arrow.up" size={20} color="#ff6b6b" />
            <Text style={styles.shareButtonText}>Share List</Text>
          </AnimatedPressable>
        </View>
      )}
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
    gap: 16,
  },
  headerTitle: {
    marginTop: 55,
    fontSize: 28,
    fontWeight: '700',
    color: '#000',
  },
  headerSubtitle: {
    fontSize: 16,
    color: '#666',
    marginTop: 4,
  },
  statsCard: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 2,
  },
  progressBar: {
    height: 8,
    backgroundColor: '#f0f0f0',
    borderRadius: 4,
    overflow: 'hidden',
  },
  progressFill: {
    height: '100%',
    backgroundColor: '#32d74b',
  },
  progressText: {
    marginTop: 8,
    color: '#666',
    fontSize: 14,
    textAlign: 'center',
  },
  emptyStateCard: {
    backgroundColor: '#fff',
    borderRadius: 16,
    padding: 24,
    alignItems: 'center',
    gap: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 3,
  },
  emptyStateTitle: {
    fontSize: 20,
    fontWeight: '700',
    color: '#000',
  },
  emptyStateText: {
    fontSize: 16,
    color: '#666',
    textAlign: 'center',
  },
  listContainer: {
    gap: 12,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: '600',
    marginBottom: 8,
  },
  itemCard: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#fff',
    padding: 16,
    borderRadius: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 2,
  },
  itemCardBought: {
    backgroundColor: '#f8f8f8',
  },
  checkbox: {
    marginRight: 12,
  },
  itemText: {
    flex: 1,
    fontSize: 16,
  },
  itemTextBought: {
    color: '#666',
    textDecorationLine: 'line-through',
  },
  actionHint: {
    fontSize: 14,
    color: '#999',
  },
  boughtSection: {
    marginTop: 24,
    gap: 12,
  },
  boughtHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  clearButton: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
  },
  clearButtonText: {
    color: '#ff6b6b',
    fontWeight: '500',
  },
  actionsContainer: {
    marginTop: 8,
  },
  shareButton: {
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
  shareButtonText: {
    color: '#ff6b6b',
    fontSize: 16,
    fontWeight: '600',
  },
});
