import React, { createContext, useContext, useState } from 'react';

export type Recipe = {
  title: string;
  ingredients: string[];
  youtubeLink?: string;
  youtubeUrl?: string;
};

type KitchenContextType = {
  recipe: Recipe | null;
  pantry: string[];
  loading: boolean;
  boughtItems: string[];
  currentVideoId: string | null;
  loadRecipeFromLink: (link: string) => Promise<void>;
  setCurrentVideoId: (id: string) => void;
  scanPantry: () => Promise<void>;
  uploadPhoto: () => Promise<void>;
  markItemBought: (item: string) => void;
  clearBoughtItems: () => void;
  addPantryItem: (item: string) => void;
};

const KitchenContext = createContext<KitchenContextType | undefined>(undefined);

export const useKitchen = () => {
  const ctx = useContext(KitchenContext);
  if (!ctx) throw new Error('useKitchen must be used within KitchenProvider');
  return ctx;
};

export const KitchenProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [recipe, setRecipe] = useState<Recipe | null>(null);
  const [pantry, setPantry] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [boughtItems, setBoughtItems] = useState<string[]>([]);
  const [currentVideoId, setCurrentVideoId] = useState<string | null>(null);

  const loadRecipeFromLink = async (link: string) => {
    setLoading(true);
    // Mock parsing YouTube link: simulate network work
    await new Promise((r) => setTimeout(r, 900));

    // Simple mock recipe data. In a real app you'd fetch/parse video or use an API.
    const mockIngredients = ['Eggs', 'Flour', 'Milk', 'Butter', 'Salt'];
    setRecipe({ title: `Recipe from ${link || 'YouTube'}`, ingredients: mockIngredients, youtubeLink: link, youtubeUrl: link });
    setBoughtItems([]);
    setLoading(false);
  };

  const scanPantry = async () => {
    setLoading(true);
    // Mock scanning - replace with real camera + vision processing
    await new Promise((r) => setTimeout(r, 1200));
    // Example: detected Eggs and Salt
    setPantry((prev) => Array.from(new Set([...prev, 'Eggs', 'Salt'])));
    setLoading(false);
  };

  const uploadPhoto = async () => {
    setLoading(true);
    await new Promise((r) => setTimeout(r, 1000));
    setPantry((prev) => Array.from(new Set([...prev, 'Milk', 'Butter'])));
    setLoading(false);
  };

  const markItemBought = (item: string) => {
    setBoughtItems((prev) => (prev.includes(item) ? prev.filter((i) => i !== item) : [...prev, item]));
  };

  const clearBoughtItems = () => setBoughtItems([]);

  const addPantryItem = (item: string) => setPantry((prev) => Array.from(new Set([...prev, item])));

  return (
    <KitchenContext.Provider
      value={{ 
        recipe, 
        pantry, 
        loading, 
        boughtItems, 
        currentVideoId,
        loadRecipeFromLink, 
        setCurrentVideoId,
        scanPantry, 
        uploadPhoto, 
        markItemBought, 
        clearBoughtItems, 
        addPantryItem 
      }}>
      {children}
    </KitchenContext.Provider>
  );
};

export default KitchenContext;
