// Kite: one full-screen button. kite://start (the Action button Shortcut) or a tap starts sending GPS to the Kite
// computer over Bluetooth; kite://stop or another tap stops. Every state change is spoken and felt.
import * as Haptics from 'expo-haptics';
import * as Speech from 'expo-speech';
import { StatusBar } from 'expo-status-bar';
import { useCallback, useEffect, useRef, useState } from 'react';
import { AccessibilityInfo, Animated, AppState, Easing, Linking, Pressable, StyleSheet, Text, View } from 'react-native';
import { type LinkState, startLink } from './ble';
import { frameFix } from './frame';
import { startLocation, stopStaleLocation } from './location';

type Mode = 'idle' | LinkState;

const C = {
  bg: '#000000',
  text: '#F2F2F7',
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
      else if (prefix) start(prefix); // opened any other way (Action button "Open App", home screen): start too
    };
    // GPS left registered by a previous run that was killed; start() registers it again
    stopStaleLocation().finally(() => Linking.getInitialURL().then((url) => handle(url, 'Kite open. ')));
    const sub = Linking.addEventListener('url', ({ url }) => handle(url));
    // Brought back to the front while stopped (Action button on a running app): start again
    const fg = AppState.addEventListener('change', (s) => {
      if (s === 'active' && !stops.current) start('Kite open. ');
    });
    return () => {
      sub.remove();
      fg.remove();
    };
  }, [start, stop]);

  const active = mode !== 'idle';
  const word = mode === 'idle' ? (stopped ? 'Stopped' : 'Ready') : WORD[mode];
  const sub = SUB[mode === 'idle' && stopped ? 'stopped' : mode];
  const spokenDetail = active ? [acc != null && `accuracy ${Math.round(acc)} metres`, `${sent} sent`].filter(Boolean).join(', ') : '';

  return (
    <Pressable
      style={styles.screen}
      onPress={() => (active ? stop() : start())}
      accessibilityRole="button"
      accessibilityLabel={`Kite. ${active ? SPOKEN[mode] : `${word}. ${sub}`}${spokenDetail ? ` ${spokenDetail}.` : ''}`}
      accessibilityHint={active ? 'Double tap to stop.' : 'Double tap to start.'}
      accessibilityState={{ selected: active, busy: mode === 'searching' || mode === 'lost' }}
    >
      {({ pressed }) => (
        <>
          <StatusBar style="light" />
          <Text style={styles.brand} maxFontSizeMultiplier={1.3}>
            Kite
          </Text>
          <View style={[styles.middle, pressed && styles.pressed]}>
            <Beacon mode={mode} />
            <Text style={styles.word} maxFontSizeMultiplier={1.4}>
              {word}
            </Text>
            <Text style={styles.sub} maxFontSizeMultiplier={1.4}>
              {sub}
            </Text>
          </View>
          <View style={[styles.card, !active && styles.cardOff]}>
            <Stat value={active && acc != null ? `±${Math.round(acc)} m` : '—'} label="Accuracy" />
            <View style={styles.divider} />
            <Stat value={active ? String(sent) : '—'} label="Sent" />
          </View>
        </>
      )}
    </Pressable>
  );
}

const SUB: Record<Mode | 'stopped', string> = {
  idle: 'Tap anywhere to begin',
  stopped: 'Tap anywhere to start again',
  searching: 'Looking for the Kite computer',
  connected: 'Sending your location',
  lost: 'Trying to reach Kite again',
};

const TINT: Record<Mode, string> = { idle: '#636366', searching: '#FFFFFF', connected: '#30D158', lost: '#FF9F0A' };

function Stat({ value, label }: { value: string; label: string }) {
  return (
    <View style={styles.stat}>
      <Text style={styles.statValue} maxFontSizeMultiplier={1.3}>
        {value}
      </Text>
      <Text style={styles.statLabel} maxFontSizeMultiplier={1.3}>
        {label.toUpperCase()}
      </Text>
    </View>
  );
}

// A ring with a dot in it. Searching: ripples spread out from the dot. Connected: green, still, softly lit.
// Lost: amber, pulsing slowly. Idle: grey.
function Beacon({ mode }: { mode: Mode }) {
  const t = useRef(new Animated.Value(0)).current;
  const reduceMotion = useReduceMotion();
  const moving = (mode === 'searching' || mode === 'lost') && !reduceMotion;
  const tint = TINT[mode];

  useEffect(() => {
    t.setValue(0);
    if (!moving) return;
    const loop = Animated.loop(
      Animated.timing(t, { toValue: 1, duration: mode === 'lost' ? 2400 : 1800, easing: Easing.out(Easing.quad), useNativeDriver: true }),
    );
    loop.start();
    return () => loop.stop();
  }, [moving, mode, t]);

  const ripple = (delay: number) => {
    const p = Animated.modulo(Animated.add(t, delay), 1);
    return {
      opacity: p.interpolate({ inputRange: [0, 1], outputRange: [0.5, 0] }),
      transform: [{ scale: p.interpolate({ inputRange: [0, 1], outputRange: [0.2, 1] }) }],
    };
  };

  return (
    <View style={styles.beacon}>
      <View style={[styles.ring, { borderColor: mode === 'idle' ? '#2C2C2E' : `${tint}33` }]} />
      {moving && <Animated.View style={[styles.ripple, { borderColor: tint }, ripple(0)]} />}
      {moving && <Animated.View style={[styles.ripple, { borderColor: tint }, ripple(0.5)]} />}
      {mode === 'connected' && <View style={[styles.glow, { backgroundColor: `${tint}26` }]} />}
      <View style={[styles.core, { backgroundColor: tint }]} />
    </View>
  );
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

const RING = 220;

const styles = StyleSheet.create({
  screen: { flex: 1, backgroundColor: C.bg, paddingTop: 64, paddingBottom: 40, paddingHorizontal: 24 },
  brand: { color: C.text, fontSize: 20, fontWeight: '600', letterSpacing: -0.4 },
  middle: { flex: 1, alignItems: 'center', justifyContent: 'center' },
  pressed: { opacity: 0.7 },
  beacon: { width: RING, height: RING, alignItems: 'center', justifyContent: 'center', marginBottom: 48 },
  ring: { position: 'absolute', width: RING, height: RING, borderRadius: RING / 2, borderWidth: 1 },
  ripple: { position: 'absolute', width: RING, height: RING, borderRadius: RING / 2, borderWidth: 1.5 },
  glow: { position: 'absolute', width: 120, height: 120, borderRadius: 60 },
  core: { width: 32, height: 32, borderRadius: 16 },
  word: { color: C.text, fontSize: 34, fontWeight: '700', letterSpacing: -0.6, textAlign: 'center' },
  sub: { color: 'rgba(235, 235, 245, 0.6)', fontSize: 17, marginTop: 8, textAlign: 'center' },
  card: { flexDirection: 'row', backgroundColor: '#1C1C1E', borderRadius: 20, paddingVertical: 18 },
  cardOff: { opacity: 0.5 },
  stat: { flex: 1, alignItems: 'center' },
  statValue: { color: C.text, fontSize: 22, fontWeight: '600', fontVariant: ['tabular-nums'] },
  statLabel: { color: 'rgba(235, 235, 245, 0.45)', fontSize: 12, fontWeight: '600', letterSpacing: 0.8, marginTop: 4 },
  divider: { width: StyleSheet.hairlineWidth, backgroundColor: '#3A3A3C', marginVertical: 4 },
});
