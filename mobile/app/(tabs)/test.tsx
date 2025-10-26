import React, { useCallback, useRef, useState, useEffect } from "react";
import { View, Button, StyleSheet, Text, Dimensions, Platform, Alert } from "react-native";
import YoutubePlayer, { YoutubeIframeRef } from "react-native-youtube-iframe";
import { useKitchen } from "@/components/KitchenContext";
import { CameraView, CameraType, useCameraPermissions } from "expo-camera";
import * as FileSystem from 'expo-file-system'; // To read file data

// --- CONFIGURATION ---
// Base URL for your FastAPI backend
const BACKEND_URL = 'http://172.20.10.6:8000'; // REPLACE WITH YOUR SERVER IP (e.g., your computer's local IP)
// Prediction rate (e.g., 6 times per second, matching your Python logic)
const PREDICTION_INTERVAL_MS = 1000 / 6;
// -----------------------

export default function App() {
  const SCREEN_H = Dimensions.get("window").height;
  const HALF = Math.round(SCREEN_H * 0.5);
  const CONTROLS_H = 48;
  const CAM_H = Math.round(SCREEN_H * 0.35);

  // YouTube State and Refs (Unchanged)
  const playerRef = useRef<YoutubeIframeRef>(null);
  const [ready, setReady] = useState(false);
  const [playing, setPlaying] = useState(false);
  const { currentVideoId } = useKitchen();
  const videoId = currentVideoId || "_Zt1EuIEhvw";
  const onReady = useCallback(() => setReady(true), []);
  const onStateChange = useCallback((state: string) => {
    if (state === "ended") setPlaying(false);
  }, []);
  const togglePlaying = useCallback(() => {
    if (!ready) return;
    setPlaying((p) => !p);
  }, [ready]);
  const seekBy = useCallback(async (deltaSeconds: number) => {
    if (!ready) return;
    const api = playerRef.current;
    if (!api?.getCurrentTime || !api?.seekTo) return;
    try {
      const cur = await api.getCurrentTime();
      const duration = await api.getDuration?.();
      let target = cur + deltaSeconds;
      if (typeof duration === "number") {
        target = Math.max(0, Math.min(duration - 0.5, target));
      } else {
        target = Math.max(0, target);
      }
      await api.seekTo(target, true);
    } catch { }
  }, [ready]);

  // --- CAMERA & GESTURE STATE ---
  const cameraRef = useRef<CameraView>(null);
  const [permission, requestPermission] = useCameraPermissions();
  const [facing] = useState<CameraType>("front");
  const [gesture, setGesture] = useState("Ready"); // State for predicted gesture
  const [isCalibrating, setIsCalibrating] = useState(true); // Matches Python state
  const predictionIntervalRef = useRef<NodeJS.Timeout | null>(null);
  // --------------------------------

  // --- GESTURE PREDICTION LOGIC ---

  const sendFrameForPrediction = async () => {
    if (!cameraRef.current) return;

    try {
      // Capture a photo (Expo's takePictureAsync provides image data as URI)
      const photo = await cameraRef.current.takePictureAsync({
        base64: false, // We'll read the URI as bytes for the FormData post
        quality: 0.5, // Lower quality to reduce network bandwidth
        // IMPORTANT: The server expects the ROI of the video feed. 
        // Ideally, you crop the frame to the green box size BEFORE sending 
        // to save bandwidth, but for simplicity here, we send the image 
        // and the server figures out the ROI.
      });

      if (!photo.uri) return;

      // Convert the image URI to a Blob or FormData for sending
      // Using fetch to get the blob/file object is often the most reliable way in Expo.
      const fileUri = photo.uri;

      // 1. Create a FormData object for the multipart/form-data request
      const formData = new FormData();

      // 2. Append the image file. The 'name' must match the FastAPI endpoint (image: UploadFile = File(...))
      // NOTE: The type must be correct (e.g., image/jpeg)
      formData.append('image', {
        uri: Platform.OS === 'android' ? fileUri : fileUri.replace('file://', ''),
        type: 'image/jpeg',
        name: 'frame.jpg',
      } as any); // Use 'any' to bypass TS errors for React Native file structure

      // 3. Append calibration status (is_calibrating: str = Form(...))
      formData.append('is_calibrating', isCalibrating ? 'true' : 'false');

      // 4. POST to FastAPI
      const response = await fetch(`${BACKEND_URL}/predict_gesture`, {
        method: 'POST',
        body: formData,
        // The 'Content-Type' header is usually automatically set to 'multipart/form-data' 
        // when using FormData, but sometimes needs to be explicit.
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      const result = await response.json();

      if (response.ok) {
        setGesture(result.gesture);
        // If the server returns a 'calibrating' status, keep the flag true
        if (result.status === "calibrating") {
          setIsCalibrating(true);
        } else {
          setIsCalibrating(false); // Calibration successful/done
        }
      } else {
        setGesture(`Error: ${result.detail || response.status}`);
        console.error("API Error:", result);
      }

    } catch (e) {
      console.error("Prediction failed:", e);
      setGesture("Prediction Failed");
    }
  };

  useEffect(() => {
    // Start the prediction interval when the component mounts and permission is granted
    if (permission?.granted) {
      predictionIntervalRef.current = setInterval(sendFrameForPrediction, PREDICTION_INTERVAL_MS);
    }

    // Cleanup: Clear the interval when the component unmounts
    return () => {
      if (predictionIntervalRef.current) {
        clearInterval(predictionIntervalRef.current);
      }
    };
  }, [permission?.granted]);


  // --- PERMISSIONS AND RENDERING ---

  if (!permission) return <View style={{ flex: 1 }} />;
  if (!permission.granted) {
    return (
      <View style={styles.centered}>
        <Text style={styles.message}>We need your permission to show the camera</Text>
        <Button onPress={requestPermission} title="Grant permission" />
      </View>
    );
  }

  const playerHeight = Math.max(120, HALF - CONTROLS_H);

  return (
    <View style={styles.screen}>
      {/* Top half: video + controls */}
      <View style={{ height: 420, paddingTop: 50, width: 400, display: "flex", justifyContent: "center" }}>
        <YoutubePlayer
          style={{ width: 400 }}
          ref={playerRef}
          height={playerHeight}
          play={playing}
          videoId={videoId}
          onReady={onReady}
          onChangeState={onStateChange}
          webViewProps={{
            allowsInlineMediaPlayback: true,
            mediaPlaybackRequiresUserAction: false,
          }}
        />
        <View style={styles.controls}>
          <Button title="-10s" onPress={() => seekBy(-10)} />
          <Button title={playing ? "pause" : "play"} onPress={togglePlaying} />
          <Button title="+10s" onPress={() => seekBy(10)} />
        </View>
      </View>

      {/* Smaller camera view below */}
      <View style={{ marginTop: 50, alignItems: 'center' }}>
        <CameraView
          ref={cameraRef}
          style={{ width: 400, height: CAM_H, borderRadius: 10, overflow: 'hidden' }}
          facing={facing}
        />
        <View style={styles.gestureOverlay}>
          <Text style={styles.gestureText}>
            {isCalibrating ? "Calibrating..." : `Gesture: ${gesture}`}
          </Text>
          <Text style={styles.instructionText}>
            {isCalibrating ? "Hold still, green box must be empty!" : "Place hand in the center."}
          </Text>
        </View>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  screen: { flex: 1, alignItems: 'center' },
  centered: { flex: 1, justifyContent: "center", alignItems: "center", padding: 16 },
  message: { textAlign: "center", paddingBottom: 0 },
  controls: {
    height: 40,
    paddingHorizontal: 30,
    paddingVertical: 0,
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
  },
  gestureOverlay: {
    position: 'absolute',
    bottom: 0,
    backgroundColor: 'rgba(0, 0, 0, 0.6)',
    width: '100%',
    padding: 10,
    alignItems: 'center',
  },
  gestureText: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#FFF',
  },
  instructionText: {
    fontSize: 14,
    color: '#DDD',
  },
});