import React, { useCallback, useRef, useState } from "react";
import { View, Button, StyleSheet, Text, Dimensions, Platform } from "react-native";
import YoutubePlayer, { YoutubeIframeRef } from "react-native-youtube-iframe";
import { useKitchen } from "@/components/KitchenContext";
import { CameraView, CameraType, useCameraPermissions } from "expo-camera";

export default function App() {
  const SCREEN_H = Dimensions.get("window").height;
  const HALF = Math.round(SCREEN_H * 0.5);

  
  // Make controls more compact and account for their height in the player height
  const CONTROLS_H = 48;

  // Reduce camera size (smaller than half screen) while keeping full frame visible
  const CAM_H = Math.round(SCREEN_H * 0.35); // ~35% of screen height

  // YouTube
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
    } catch {}
  }, [ready]);

  // Camera permissions
  const [permission, requestPermission] = useCameraPermissions();
  const [facing] = useState<CameraType>("front");

  if (!permission) return <View style={{ flex: 1 }} />;
  if (!permission.granted) {
    return (
      <View style={styles.centered}>
        <Text style={styles.message}>We need your permission to show the camera</Text>
        <Button onPress={requestPermission} title="Grant permission" />
      </View>
    );
  }

  // Player height accounts for top padding + controls, so no extra gap
  const playerHeight = Math.max(120, HALF - CONTROLS_H);

  return (
    <View style={styles.screen}>
      {/* Top half: video + controls */}
      <View style={{ height: 420, paddingTop: 50, width:400, display:"flex", justifyContent:"center"}}>
        <YoutubePlayer
          style={{width: 400}}
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

      {/* Smaller camera view below; set a fixed height and a common ratio to avoid cropping */}
      <CameraView
        style={{ width: "auto", height: CAM_H, marginTop: 50 }}
        facing={facing}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  screen: { flex: 1 },
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
});