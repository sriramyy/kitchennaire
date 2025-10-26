import cv2
import numpy as np
import os
import time

# Use the compatible imports for your local environment
from tensorflow.keras.models import load_model 
from tensorflow.keras.layers import BatchNormalization

# Global variable to store the running average of the background
bg = None

# --- Helper Functions (Background Subtraction) ---

def run_avg(image, accumWeight):
    """Accumulates a weighted average of the background."""
    global bg
    if bg is None:
        bg = image.copy().astype("float")
        return
    # Accumulate the weighted average, which adapts to slow changes
    cv2.accumulateWeighted(image, bg, accumWeight)

def segment(image, threshold=15): # ADJUSTED THRESHOLD (Lowered for better hand coverage)
    """Segments the hand region from the background."""
    global bg
    
    # Calculate the absolute difference between the background and the current frame
    diff = cv2.absdiff(bg.astype("uint8"), image)
    
    # Threshold the difference image, creating a binary mask of the foreground (hand)
    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]
    
    # Find contours on the thresholded image
    (cnts, _) = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # If no contours were found, return None
    if len(cnts) == 0:
        return None
    else:
        # Get the largest contour (assumed to be the hand)
        segmented = max(cnts, key=cv2.contourArea)
        # Return the binary image and the segmented contour
        return (thresholded, segmented)


# --- Model Weights and Prediction Functions ---

# load Model Weights
def _load_weights():
    # Load the NEW, compatible model file with custom objects for BatchNormalization
    custom_objects = {'BatchNormalization': BatchNormalization}
    try:
        model = load_model("hand_gesture_recognition.h5", custom_objects=custom_objects)
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"[ERROR] Model loading failed: {e}")
        return None

    
def getPredictedClass(model):
    # This function uses the Temp.png image saved by the main loop
    image = cv2.imread('Temp.png')
    
    # Ensure image loaded before processing
    if image is None:
        return "Waiting..."

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.resize(gray_image, (100, 120))

    gray_image = gray_image.reshape(1, 100, 120, 1)

    prediction = model.predict_on_batch(gray_image)

    predicted_class = np.argmax(prediction)
    if predicted_class == 0:
        return "Blank"
    elif predicted_class == 1:
        return "OK"
    elif predicted_class == 2:
        return "Thumbs Up"
    elif predicted_class == 3:
        return "Thumbs Down"
    elif predicted_class == 4:
        return "Punch"
    elif predicted_class == 5:
        return "High Five"
    else:
        return "Unknown"


# --- Main Live Execution Loop ---

if __name__ == "__main__":
    # initialize accumulated weight
    accumWeight = 0.5

    # get the reference to the webcam (usually 0)
    camera = cv2.VideoCapture(0)

    # Check if camera opened successfully
    if not camera.isOpened():
        print("[FATAL] Could not open webcam. Ensure no other apps are using it.")
        exit()

    fps = int(camera.get(cv2.CAP_PROP_FPS))
    
    # region of interest (ROI) coordinates (top, right, bottom, left)
    # INCREASED SIZE for easier tracking:
    top, right, bottom, left = 10, 350, 225, 590
    
    # initialize num of frames
    num_frames = 0
    
    model = _load_weights()
    if model is None:
        camera.release()
        cv2.destroyAllWindows()
        exit()
        
    k = 0
    print("-" * 40)
    print("Press 'q' to exit the application.")
    print("Ensure the green box is empty during calibration.")
    print("-" * 40)
    
    # keep looping, until interrupted
    while (True):
        # get the current frame
        (grabbed, frame) = camera.read()
        
        if not grabbed:
            print("[ERROR] Failed to grab frame.")
            break

        # resize the frame and flip it (to avoid mirror effect)
        frame = cv2.resize(frame, (700,700))
        frame = cv2.flip(frame, 1)

        clone = frame.copy()

        # get the ROI
        roi = frame[top:bottom, right:left]

        # convert the roi to grayscale and blur it
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (9, 9), 0) # Increased blur kernel (9, 9)

        # Calibration phase (first 30 frames)
        if num_frames < 30:
            run_avg(gray, accumWeight)
            cv2.putText(clone, "CALIBRATING...", (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            if num_frames == 1:
                print("[STATUS] please wait! calibrating...")
            elif num_frames == 29:
                print("[STATUS] calibration successfull! Ready for gestures.")
        else:
            # Segmentation phase
            hand = segment(gray)

            if hand is not None:
                (thresholded, segmented) = hand

                # Draw the segmented contour on the video feed
                cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255), 2)

                # Run prediction only periodically
                if k % (fps / 6) == 0:
                    cv2.imwrite('Temp.png', thresholded) # Save the image for prediction
                    predictedClass = getPredictedClass(model)
                
                # Display the predicted class
                try:
                    cv2.putText(clone, str(predictedClass), (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                except NameError:
                    # Initial state before first prediction
                    cv2.putText(clone, "Ready", (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


                # show the thresholded image
                cv2.imshow("Thresholded Hand Segment", thresholded)
        
        k = k + 1
        
        # Draw the ROI rectangle on the main feed
        cv2.rectangle(clone, (left, top), (right, bottom), (0, 255, 0), 2)

        # increment the number of frames
        num_frames += 1

        # display the frame with segmented hand
        cv2.imshow("Video Feed - Hand Gesture Recognizer", clone)

        # observe the keypress by the user
        keypress = cv2.waitKey(1) & 0xFF

        # if the user pressed "q", then stop looping
        if keypress == ord("q"):
            break

    # free up memory
    camera.release()
    cv2.destroyAllWindows()