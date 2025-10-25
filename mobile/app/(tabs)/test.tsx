import React, { useCallback, useRef, useState } from "react";
import { View, Button } from "react-native";
import YoutubePlayer, { YoutubeIframeRef } from "react-native-youtube-iframe";

export default function App() {
  const playerRef = useRef<YoutubeIframeRef>(null);
  const [ready, setReady] = useState(false);
  const [playing, setPlaying] = useState(false);

  const onReady = useCallback(() => setReady(true), []);
  const onStateChange = useCallback((state: string) => {
    if (state === "ended") setPlaying(false);
  }, []);

  const togglePlaying = useCallback(() => {
    if (!ready) return;
    setPlaying(p => !p);
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
        target = Math.max(0, Math.min(duration - 0.5, target)); // clamp
      } else {
        target = Math.max(0, target);
      }
      await api.seekTo(target, true);
    } catch {}
  }, [ready]);

  return (
    <View style={{ gap: 12, padding: 16 }}>
      <YoutubePlayer
        ref={playerRef}
        height={300}
        play={playing}
        videoId="_Zt1EuIEhvw"
        onReady={onReady}
        onChangeState={onStateChange}
      />
      <View style={{ gap: 8 }}>
        <Button title="-10s" onPress={() => seekBy(-10)} />
        <Button title={playing ? "pause" : "play"} onPress={togglePlaying} />
        <Button title="+10s" onPress={() => seekBy(10)} />
      </View>
    </View>
  );
}