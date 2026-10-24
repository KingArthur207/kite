// Kite: one full-screen button. kite://start (the Action button Shortcut) or a tap starts sending GPS to the Kite
// computer over Bluetooth; kite://stop or another tap stops. Every state change is spoken and felt.
import * as Haptics from 'expo-haptics';
import * as Speech from 'expo-speech';
import { StatusBar } from 'expo-status-bar';
import { useCallback, useEffect, useRef, useState } from 'react';
import { AccessibilityInfo, Animated, Easing, Linking, Pressable, StyleSheet, Text, View } from 'react-native';
import { type LinkState, startLink } from './ble';
import { frameFix } from './frame';
import { startLocation, stopStaleLocation } from './location';

type Mode = 'idle' | LinkState;

const C = {
  bg: '#000000',
  text: '#F2F2F7',
  dot: '#FFFFFF',
  dim: '#48484A',
  footer: 'rgba(235, 235, 245, 0.4)',
};

const SPOKEN: Record<Mode, string> = {
  idle: 'Stopped.',
  searching: 'Looking for Kite.',
  connected: 'Connected. Sending your location.',
  lost: 'Lost connection. Reconnecting.',
};

const WORD: Record<LinkState, string> = { searching: 'Searching', connected: 'Connected', lost: 'Reconnecting' };

function say(text: string): void {
  Speech.stop();
  // false: the system's own audio session, so speech is heard with the ringer switch on silent
  Speech.speak(text, { useApplicationAudioSession: false });
}

const feel: Record<Mode, () => void> = {
  idle: () => Haptics.notificationAsync(Haptics.NotificationFeedbackType.Warning),
  searching: () => Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium),
  connected: () => Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success),
  lost: () => Haptics.notificationAsync(Haptics.NotificationFeedbackType.Error),
};

function announce(mode: Mode, prefix = ''): void {
  say(prefix + SPOKEN[mode]);
  feel[mode]();
}

export default function App() {
  const [mode, setMode] = useState<Mode>('idle');
  const [stopped, setStopped] = useState(false);
  const [acc, setAcc] = useState<number | null>(null);
  const [sent, setSent] = useState(0);
  const modeRef = useRef<Mode>('idle');
  const stops = useRef<(() => void)[] | null>(null);
  const line = useRef<string | null>(null);

  const go = useCallback((next: Mode, prefix = '') => {
    if (stops.current === null && next !== 'idle') return; // a late event after stop
    modeRef.current = next;
    setMode(next);
    announce(next, prefix);
  }, []);

  const stop = useCallback(() => {
    if (!stops.current) return;
    stops.current.forEach((fn) => fn());
    stops.current = null;
    line.current = null;
    setAcc(null);
    setStopped(true);
    go('idle');
  }, [go]);

  const start = useCallback(
    (prefix = '') => {
      if (stops.current) return announce(modeRef.current, prefix); // already on: say where we are
      const list: (() => void)[] = [];
      stops.current = list;
      setSent(0);
      go('searching', prefix);
      list.push(startLink({ nextLine: () => line.current, onState: (s) => go(s), onSent: setSent }));
      startLocation((fix) => {
        line.current = frameFix(fix);
        setAcc(fix.acc);
      })
        .then((stopLoc) => (stops.current === list ? list.push(stopLoc) : stopLoc()))
        .catch((e) => {
          console.warn('[kite] location', e);
          stop();
          say('Kite needs your location. Allow it in Settings, then try again.');
        });
    },
    [go, stop],
  );

  // kite://start and kite://stop, both on a cold launch and while running
  useEffect(() => {
    const handle = (url: string | null | undefined, prefix = '') => {
      const cmd = url?.replace(/^kite:\/\//i, '').split(/[/?#]/)[0].toLowerCase();
      if (cmd === 'start') start(prefix);
      else if (cmd === 'stop') stop();
      else if (prefix) say(prefix.trim());
    };
    // GPS left registered by a previous run that was killed; start() registers it again
    stopStaleLocation().finally(() => Linking.getInitialURL().then((url) => handle(url, 'Kite open. ')));
    const sub = Linking.addEventListener('url', ({ url }) => handle(url));
    return () => sub.remove();
  }, [start, stop]);

  const active = mode !== 'idle';
  const word = mode === 'idle' ? (stopped ? 'Stopped' : 'Tap to start') : WORD[mode];
  const detail = active ? [acc != null && `±${Math.round(acc)} m`, `${sent} sent`].filter(Boolean).join('  ·  ') : '';
  const spokenDetail = active ? [acc != null && `accuracy ${Math.round(acc)} metres`, `${sent} sent`].filter(Boolean).join(', ') : '';

  return (
    <Pressable
      style={styles.screen}
      onPress={() => (active ? stop() : start())}
      accessibilityRole="button"
      accessibilityLabel={`Kite. ${active ? SPOKEN[mode] : `${word}.`}${spokenDetail ? ` ${spokenDetail}.` : ''}`}
      accessibilityHint={active ? 'Double tap to stop.' : 'Double tap to start.'}
      accessibilityState={{ selected: active, busy: mode === 'searching' || mode === 'lost' }}
    >
      {({ pressed }) => (
        <>
          <StatusBar style="light" />
          <View style={[styles.middle, pressed && styles.pressed]}>
            <Dot mode={mode} />
            <Text style={styles.word} maxFontSizeMultiplier={1.5}>
              {word}
            </Text>
          </View>
          <Text style={styles.footer} maxFontSizeMultiplier={1.4}>
            {detail}
          </Text>
        </>
      )}
    </Pressable>
  );
}

// Idle: a dim dot. Searching: the white dot breathes. Connected: solid and still. Lost: a hollow ring.
function Dot({ mode }: { mode: Mode }) {
  const breath = useRef(new Animated.Value(1)).current;
  const reduceMotion = useReduceMotion();
  const breathing = mode === 'searching' && !reduceMotion;

  useEffect(() => {
    breath.setValue(1);
    if (!breathing) return;
    const half = { duration: 1000, easing: Easing.inOut(Easing.sin), useNativeDriver: true };
    const loop = Animated.loop(
      Animated.sequence([
        Animated.timing(breath, { ...half, toValue: 0.25 }),
        Animated.timing(breath, { ...half, toValue: 1 }),
      ]),
    );
    loop.start();
    return () => loop.stop();
  }, [breathing, breath]);

  const look = mode === 'idle' ? styles.dim : mode === 'lost' ? styles.ring : styles.solid;
  return <Animated.View style={[styles.dot, look, { opacity: breath }]} />;
}

function useReduceMotion(): boolean {
  const [reduce, setReduce] = useState(false);
  useEffect(() => {
    AccessibilityInfo.isReduceMotionEnabled().then(setReduce);
    const sub = AccessibilityInfo.addEventListener('reduceMotionChanged', setReduce);
    return () => sub.remove();
  }, []);
  return reduce;
}

const DOT = 14;

const styles = StyleSheet.create({
  screen: { flex: 1, backgroundColor: C.bg, paddingBottom: 44 },
  middle: { flex: 1, alignItems: 'center', justifyContent: 'center' },
  pressed: { opacity: 0.6 },
  dot: { width: DOT, height: DOT, borderRadius: DOT / 2, marginBottom: 36 },
  solid: { backgroundColor: C.dot },
  dim: { backgroundColor: C.dim },
  ring: { borderWidth: 1.5, borderColor: C.dot },
  word: { color: C.text, fontSize: 24, fontWeight: '500', letterSpacing: -0.3, textAlign: 'center' },
  footer: { color: C.footer, fontSize: 13, textAlign: 'center', fontVariant: ['tabular-nums'], minHeight: 18 },
});
