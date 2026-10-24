# How Kite *works.*

you learn it in a minute

Tap any sound below to hear it, rendered by the real earpiece. Then see why a beep beats a sentence, how you talk to Kite, and what the earpiece said on the demo walk.

## Every sound means one thing.

Where a sound comes from is where the thing is. What it sounds like is what it is. How fast it repeats is how soon you'd reach it.

where

### Direction

Rendered with a model of the human head, so each ear hears it a fraction of a millisecond apart.

what

### A sound per thing

A soft knock, a rough buzz, a clean pip, a falling glide. Never more than two at once.

how soon

### Getting closer

Beeps start about three and a half seconds before you'd meet, then speed up and rise in pitch. At the last second Kite says "Stop."

which way

### The beacon

A tick from the direction of your next turn. Bright when you're facing it.

## Inside or out, it sees for you.

Where you are decides what Kite uses. The earpiece and the voice are the same either way.

### Indoors

the chest camera alone

- Beeps for the chair, the door frame, the sign at head height.
- Point it at a **menu, a book, a street sign or a screen** and say "read this". Kite reads it to you, line by line.
- "What's in front of me?" gets a sentence, not a list of boxes.

For anywhere a drone can't fly.

### Outdoors

the drone, plus the chest camera

- The drone watches the route, the crowd at the corner and the car turning in.
- A beacon ticks toward your next turn; directions arrive on time or not at all.
- At a hazard Kite stops you, waits, and tells you when the way is clear.

For walking somewhere you need to go.

## The beep is the interface.

Speech is slow. Two words take most of a second, and the person stepping off the curb doesn't have one. So Kite saves its voice for decisions and lets sound carry the distance.

Each object gets a time-to-contact, worked out from how fast the gap is closing. The rate of the beeps follows it, from one a second at the edge of the warning zone to almost nine a second just before contact. Your ears read rhythm faster than any sentence.

Faster means closer. That's the whole manual.

_Seconds between beeps, from the earpiece's sonar settings (1.0 s → 0.11 s)._

## Just say it.

Press an AirPod once and talk. Simple requests run on the laptop with no internet at all; everything else goes to Claude, which sees what Kite sees, and comes back in Cartesia's voice.

1

**Walker:** Kite, take me to my usual coffee shop.
**Kite:** Okay, heading to Blue Bottle Coffee.

_Moss knows “usual”_

2

**Walker:** Bring me to the end of the street.
**Kite:** Okay, I'll guide you to the end of the street.

_works with no internet_

3

**Walker:** Read this to me.
**Kite:** Lunch menu. Margherita pizza, 14 dollars. Caesar salad, 11 dollars.

_Claude reads, Cartesia speaks_

4

**Walker:** What's around me?
**Kite:** A person slightly left, 2 meters; a car at 2 o'clock, 8 meters.

_from what the cameras see_

5

**Walker:** Remember this place as the bench.
**Kite:** Saved. I'll remember this place as the bench.

_Moss, again_

6

**Walker:** *(walks toward someone)*
**Kite:** Stop. Person ahead.

_plain code, no AI, instant_

## What the earpiece *actually said.*

The log from the 41-second walk on the home page, verbatim. Every line has a priority; a "Stop." cuts in front of everything else.

**python -m earpiece run --sim walk

```text

000.5s  said [nav] Head north on Lyon Street for 60 meters.
006.5s  said [hazard] Person, ahead.
008.4s  said [critical] Stop. Person ahead.
013.4s  said [hazard] Obstacle, ahead.
015.5s  said [critical] Stop. Obstacle ahead.
018.0s  said [hazard] Don't walk.
022.5s  said [hazard] Walk signal is on.
025.0s  said [nav] Turn left onto Bay Street.
026.8s  said [nav] Small crowd ahead on the right.
030.0s  said [system] Drone battery is at 28 percent.
033.1s  said [hazard] Heads up: The curb at Lyon Street and Bay Street is very tall: step down slowly

```

---

Kite · https://trykite.pages.dev/how · Pages: [Home](https://trykite.pages.dev/) · [How it works](https://trykite.pages.dev/how) · [For judges](https://trykite.pages.dev/judges) · [About](https://trykite.pages.dev/about) · [Contact](https://trykite.pages.dev/contact) · [Privacy](https://trykite.pages.dev/privacy) · [llms.txt](https://trykite.pages.dev/llms.txt) · [Sitemap](https://trykite.pages.dev/sitemap.xml)
