import React, { useCallback, useRef, useState, useEffect } from "react";
// Standard React Native imports
import { View, Button, StyleSheet, Text, Dimensions, Platform, Alert } from "react-native";
// Assuming necessary components are available locally or correctly mapped
import YoutubePlayer, { YoutubeIframeRef } from "react-native-youtube-iframe";
// Make sure this path is correct for your project structure
import { useKitchen } from "../../components/KitchenContext";

// --- CONFIGURATION ---
// Base URL for your FastAPI backend running alongside the webcam script
const BACKEND_URL = 'http://172.20.10.6:8000'; // REPLACE WITH YOUR SERVER IP
// How often to fetch the latest gesture from the backend (e.g., every 500ms)
const GESTURE_POLL_INTERVAL_MS = 500;
// -----------------------

export default function App() {
  const SCREEN_H = Dimensions.get("window").height;
  const HALF = Math.round(SCREEN_H * 0.5);
  const CONTROLS_H = 48;
  const CAM_W = 400; // Keep width consistent for layout

  // YouTube State and Refs
  const playerRef = useRef<YoutubeIframeRef>(null);
  const [ready, setReady] = useState(false);
  const [playing, setPlaying] = useState(false);
  const { currentVideoId } = useKitchen();
  const videoId = currentVideoId || "_Zt1EuIEhvw"; // Default video ID

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

  // --- GESTURE STATE ---
  const [gesture, setGesture] = useState("Connecting..."); // State for predicted gesture fetched from backend
  const pollIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const isMounted = useRef(false); // Track component mount status
  const [isBackendCalibrating, setIsBackendCalibrating] = useState(true); // Track backend calibration state via polling
  // ---------------------

  // --- GESTURE POLLING LOGIC ---
  const fetchLatestGesture = async () => {
    if (!isMounted.current) { // Safety check
      if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);
      return;
    }

    try {
      // Fetch the latest gesture from the new backend endpoint
      const response = await fetch(`${BACKEND_URL}/get_latest_gesture`);
      if (!response.ok) {
        // Handle potential server errors (e.g., 503 if model isn't loaded)
        setGesture(`Server Error: ${response.status}`);
        setIsBackendCalibrating(true); // Assume calibrating or error state
        console.error(`API Error: ${response.status}`);
        return; // Stop processing this poll attempt
      }
      const result = await response.json();

      setGesture(result.gesture); // Update gesture state

      // Check if the backend is still calibrating based on the text
      // Updated check to be more robust
      const gestureText = result.gesture || "";
      if (gestureText && typeof gestureText === 'string' &&
        (gestureText.toLowerCase().includes("calibrating") || gestureText.toLowerCase() === "initializing...")) {
        setIsBackendCalibrating(true);
      } else {
        setIsBackendCalibrating(false);
      }

    } catch (e) {
      console.error("Failed to fetch gesture:", e);
      setGesture("Network Error"); // Indicate a connection problem
      setIsBackendCalibrating(true); // Assume calibrating/error if connection fails
    }
  };

  useEffect(() => {
    isMounted.current = true;

    // Start polling for gestures when component mounts
    pollIntervalRef.current = setInterval(fetchLatestGesture, GESTURE_POLL_INTERVAL_MS);

    // Cleanup: Clear the interval when the component unmounts
    return () => {
      isMounted.current = false;
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
        pollIntervalRef.current = null;
      }
    };
  }, []); // Run only on mount and unmount

  // --- YOUTUBE GESTURE CONTROL ---
  useEffect(() => {
    // Only control video if playing, ready, and backend is NOT calibrating
    if (!playing || !ready || isBackendCalibrating) return;

    const controlVideo = (gestureCmd) => {
      const api = playerRef.current;
      if (!api) return; // Guard clause

      // Map gesture text to player actions
      switch (gestureCmd) {
        case "Thumbs Up": setPlaying(true); break;  // Play
        case "Thumbs Down": setPlaying(false); break; // Pause
        case "Punch": seekBy(10); break;       // Seek forward
        case "OK": seekBy(-10); break;      // Seek backward
        // Add cases for "High Five" or "Blank" if needed
        default: break; // Do nothing for "Blank", "Unknown", errors, etc.
      }
    };

    controlVideo(gesture); // Call control function with the latest fetched gesture

  }, [gesture, ready, playing, isBackendCalibrating, seekBy]); // Dependencies


  // --- RENDERING (No Camera, Just Display) ---

  const playerHeight = Math.max(120, HALF - CONTROLS_H);

  return (
    <View style={styles.screen}>
      {/* Top half: video + controls */}
      <View style={styles.videoContainer}>
        <YoutubePlayer
          style={{ width: '100%' }} // Use percentage for better fit
          ref={playerRef}
          height={playerHeight}
          play={playing}
          videoId={videoId}
          onReady={onReady}
          onChangeState={onStateChange}
          webViewProps={{
            allowsInlineMediaPlayback: true,
            mediaPlaybackRequiresUserAction: false, // Allows programmatic play/pause
          }}
        />
        <View style={styles.controls}>
          <Button title="-10s" onPress={() => seekBy(-10)} />
          <Button title={playing ? "pause" : "play"} onPress={togglePlaying} />
          <Button title="+10s" onPress={() => seekBy(10)} />
        </View>
      </View>

      {/* Gesture Display Area Below Video */}
      <View style={styles.gestureDisplayArea}>
        <Text style={styles.gestureText}>
          {/* Display calibration status or the fetched gesture */}
          {isBackendCalibrating ? "Calibrating..." : `Gesture: ${gesture}`}
        </Text>
        <Text style={styles.instructionText}>
          {isBackendCalibrating ? "Waiting for backend webcam..." : "Control video with hand gestures!"}
        </Text>
      </View>
    </View>
  );
}

// --- STYLES --- (Removed CameraView and ROI styles)
const styles = StyleSheet.create({
  screen: {
    flex: 1,
    alignItems: 'center',
    paddingTop: Platform.OS === 'android' ? 25 : 50, // Adjust top padding for status bar
    backgroundColor: '#f0f0f0', // Light background for the screen
  },
  centered: { // For permission request screen
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    padding: 16
  },
  message: { // For permission request text
    textAlign: "center",
    paddingBottom: 10,
    fontSize: 16,
  },
  videoContainer: { // Container for player and controls
    width: '95%', // Use percentage width
    maxWidth: 500, // Max width for larger screens
    marginBottom: 30, // Space below video
    backgroundColor: '#fff', // White background for player area
    borderRadius: 10,
    paddingTop: 10, // Padding inside the container
    elevation: 3, // Shadow for Android
    shadowColor: '#000', // Shadow for iOS
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
  },
  controls: {
    height: 48,
    paddingHorizontal: 20, // Adjusted padding
    flexDirection: "row",
    justifyContent: "space-around", // Space out buttons evenly
    alignItems: "center",
    borderTopWidth: 1, // Separator line
    borderTopColor: '#eee',
  },
  gestureDisplayArea: { // Area to show the fetched gesture
    marginTop: 20, // Adjusted spacing
    paddingVertical: 15,
    paddingHorizontal: 20,
    backgroundColor: '#444', // Darker background for contrast
    borderRadius: 8,
    alignItems: 'center',
    width: '90%', // Make it wide
    maxWidth: 500,
    elevation: 2,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.1,
    shadowRadius: 3,
  },
  gestureText: {
    fontSize: 26, // Slightly smaller font
    fontWeight: 'bold',
    color: '#FFF', // White text
    marginBottom: 5,
  },
  instructionText: {
    fontSize: 14, // Smaller instruction text
    color: '#ccc', // Lighter gray text
  },
});

