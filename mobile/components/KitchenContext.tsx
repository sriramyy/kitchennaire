// KitchenContext.tsx
import React, { createContext, useContext, useState } from 'react';

// ------------------------------------------------------------------
// TYPES for structured recipe data
export type Ingredient = {
  name: string;
  isAvailable: boolean; // true for checkmark, false for 'x'
};

// This Recipe type now matches the Python 'RecipeResponse' model
export type Recipe = {
  title: string;
  timeInMinutes: number; // Time (in minutes)
  videoDuration: string; // Video Duration (e.g., "12:45")
  ingredients: Ingredient[]; // List of structured ingredients
  video_id?: string; // The video ID from the backend
  youtubeLink?: string; // We'll add this manually
  youtubeUrl?: string; // We'll add this manually
};
// ------------------------------------------------------------------

type KitchenContextType = {
  recipe: Recipe | null;
  pantry: string[];
  loading: boolean;
  boughtItems: string[];
  currentVideoId: string | null;
  loadRecipeFromLink: (link: string) => Promise<void>; // This will now throw errors
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

// --- THIS IS THE IP FROM YOUR recipe.tsx ---
// Make sure this is your computer's LAN IP, not localhost
const API_BASE = process.env.EXPO_PUBLIC_API_BASE;

export const KitchenProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [recipe, setRecipe] = useState<Recipe | null>(null);
  const [pantry, setPantry] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [boughtItems, setBoughtItems] = useState<string[]>([]);
  const [currentVideoId, setCurrentVideoId] = useState<string | null>(null);

  const loadRecipeFromLink = async (link: string) => {
    setLoading(true);
    setRecipe(null); // Clear the old recipe

    // We wrap this in a try/catch so the UI can catch the error
    try {
      const res = await fetch(`${API_BASE}/get_recipe`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
        },
        body: JSON.stringify({ yt_url: link }),
      });

      if (!res.ok) {
        // Get the error detail from the FastAPI server
        const errData = await res.json();
        throw new Error(errData.detail || 'Failed to fetch recipe from server');
      }

      const data: Recipe = await res.json();

      // --- SET REAL DATA FROM PYTHON ---
      setRecipe({
        ...data,
        youtubeLink: link, // Add the original link back in
        youtubeUrl: link,
      });
      // ---------------------------------

      if (data.video_id) {
        setCurrentVideoId(data.video_id);
      }

      setBoughtItems([]);
    } catch (error: any) {
      console.error('Error in loadRecipeFromLink:', error);
      // Re-throw the error so the UI component (recipe.tsx) can catch it
      // and set the error message
      throw error;
    } finally {
      setLoading(false);
    }
  };
  // --- END OF UPDATED FUNCTION ---


  const scanPantry = async () => {
    setLoading(true);
    // Mock scanning - replace with real camera + vision processing
    await new Promise((r) => setTimeout(r, 1200));
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