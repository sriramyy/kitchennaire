import cv2
import numpy as np
from keras.models import load_model
from keras.layers import BatchNormalization # Import BatchNormalization
import os

from tensorflow.keras.models import load_model 
from tensorflow.keras.layers import BatchNormalization

# --- Model Loading Function ---

bg = None

def _load_weights():
    # Pass BatchNormalization as a custom object to ensure correct loading
    custom_objects = {'BatchNormalization': BatchNormalization}
    try:
        # Use the correct file name (hand_gesture_recognition.h5)
        model = load_model("hand_gesture_recognition.h5", custom_objects=custom_objects)
        
        # Print the summary to verify the model loaded correctly
        # The output shapes should match your notebook: (None, 98, 118, 32), etc.
        model.summary() 
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure 'hand_gesture_recognition.h5' is in the same directory.")
        return None

# The getPredictedClass function remains the same
def getPredictedClass(model, image_path='Temp.png'):
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not load image at {image_path}")
        
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_image = cv2.resize(gray_image, (100, 120))

        # IMPORTANT: This reshape is correct for the 2D CNN model's input
        gray_image = gray_image.reshape(1, 100, 120, 1)

        prediction = model.predict_on_batch(gray_image)

        predicted_class_index = np.argmax(prediction)
        
        if predicted_class_index == 0:
            return "Blank"
        elif predicted_class_index == 1:
            return "OK"
        elif predicted_class_index == 2:
            return "Thumbs Up"
        elif predicted_class_index == 3:
            return "Thumbs Down"
        elif predicted_class_index == 4:
            return "Punch"
        elif predicted_class_index == 5:
            return "High Five"
        else:
            return f"Unknown Prediction Index: {predicted_class_index}"

    except Exception as e:
        return f"ERROR during prediction: {e}"

# --- Main Test Execution (Modified to remove webcam) ---

if __name__ == "__main__":
    
    # Check if a sample image exists before running
    if not os.path.exists('Temp.png'):
        print("Error: 'Temp.png' not found. Create a sample image for the test.")
        exit()
        
    model = _load_weights()
    
    if model:
        predicted_gesture = getPredictedClass(model, 'Temp.png')
        
        print("\n" + "=" * 40)
        print(f"Prediction Complete.")
        print(f"Test Image 'Temp.png' Predicted as: {predicted_gesture}")
        print("=" * 40)
    else:
        print("Cannot run prediction test without a loaded model.")