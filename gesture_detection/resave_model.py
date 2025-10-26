import tensorflow as tf
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model as keras_load_model # Use full path for safety

# Define the model architecture EXACTLY as it was trained
def create_original_model_architecture():
    # Input shape matches the (100, 120, 1) image size
    model = Sequential()
    
    # First Block
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(100, 120, 1))) 
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    # Second Block
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    # Classifier Block
    model.add(Flatten())
    model.add(Dense(128, activation='relu')) # Hidden Dense Layer
    model.add(Dropout(0.5))
    
    # Output Layer (6 classes: Blank, OK, Thumbs Up, Thumbs Down, Punch, High Five)
    model.add(Dense(6, activation='softmax'))

    # Compile the model with the original optimizer/loss (important for saving the structure)
    optimiser = Adam() 
    model.compile(optimizer=optimiser, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    
    return model

# Main function to load weights and save the new model file
if __name__ == "__main__":
    original_model_path = "hand_gesture_recognition.h5"
    new_model_path = "new_gesture_model.h5"
    
    if not os.path.exists(original_model_path):
        print(f"Error: Original model file '{original_model_path}' not found in the directory.")
    else:
        try:
            # 1. Create the new, compatible model structure
            new_model = create_original_model_architecture()
            
            # 2. Load ONLY the weights from your Kaggle file (THIS IS THE FIX)
            # This bypasses the structural loading error and only transfers the learned numbers.
            new_model.load_weights(original_model_path)
            
            # 3. Save the model in the current environment's format
            new_model.save(new_model_path, save_format='h5') # Explicitly use h5 format for certainty
            
            print("\n" + "=" * 60)
            print(f"SUCCESS: Model architecture reconstructed and re-saved.")
            print(f"New, compatible file created: '{new_model_path}'")
            print("=" * 60)

        except Exception as e:
            print(f"An error occurred during the re-save process: {e}")
            print("Ensure your original H5 file is intact and that TensorFlow/Keras are installed.")