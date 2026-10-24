// GPS for Kite. A TaskManager location task keeps fixes coming with the screen locked (plain watchPositionAsync
// stops in the background on iOS); watchPositionAsync is the fallback if the task cannot start.
import * as Location from 'expo-location';
import * as TaskManager from 'expo-task-manager';
import { type Fix, type RawLocation, toFix } from './frame';

const TASK = 'kite-location';
let listener: ((fix: Fix) => void) | null = null;

function deliver(locations: RawLocation[]): void {
  const last = locations[locations.length - 1];
  if (last && listener) listener(toFix(last));
}

// Must run at module load (before the app renders), so iOS can relaunch the task.
TaskManager.defineTask<{ locations: RawLocation[] }>(TASK, async ({ data, error }) => {
  if (error) console.warn('[kite] location task', error.message);
  else if (data) deliver(data.locations);
});

const OPTIONS = {
  accuracy: Location.Accuracy.BestForNavigation,
  timeInterval: 500,
  distanceInterval: 0,
};

// iOS keeps a location task registered across launches; drop one a killed run left behind.
export async function stopStaleLocation(): Promise<void> {
  try {
    if (await Location.hasStartedLocationUpdatesAsync(TASK)) await Location.stopLocationUpdatesAsync(TASK);
  } catch (e) {
    console.warn('[kite] stale location task', e);
  }
}

// Starts GPS and calls onFix for every new fix. Resolves to a stop function; throws if permission is refused.
export async function startLocation(onFix: (fix: Fix) => void): Promise<() => void> {
  const fg = await Location.requestForegroundPermissionsAsync();
  if (!fg.granted) throw new Error('Location permission refused');
  // "Always" is nice to have; the task also runs on "While Using" because it starts in the foreground.
  await Location.requestBackgroundPermissionsAsync().catch(() => undefined);
  listener = onFix;
  try {
    await Location.startLocationUpdatesAsync(TASK, {
      ...OPTIONS,
      activityType: Location.ActivityType.OtherNavigation,
      pausesUpdatesAutomatically: false,
      showsBackgroundLocationIndicator: true,
    });
    return () => {
      listener = null;
      Location.stopLocationUpdatesAsync(TASK).catch(() => undefined);
    };
  } catch (e) {
    console.warn('[kite] background location unavailable, foreground only', e);
    const sub = await Location.watchPositionAsync(OPTIONS, (loc) => deliver([loc]));
    return () => {
      listener = null;
      sub.remove();
    };
  }
}
