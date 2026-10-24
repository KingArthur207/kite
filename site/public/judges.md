# Kite *in 30 seconds.*

for the people with the clipboards

**Who it's for**

Blind and low-vision people who walk city streets on their own: to the coffee shop, the pharmacy, the bus stop.

**The problem**

About a million Americans are blind and fewer than 10,000 guide dogs work in the US. A dog costs $50,000 and the wait is about a year.

**What Kite does**

A DJI Tello flies 30 cm above and behind your head; a chest camera watches your feet. AirPods beep from where things are, faster as they get closer, and a calm voice says "Stop."

**Try it at our table**

Put on a sleep mask and the AirPods and walk. When one of us steps in front of you, the beeps speed up and Kite says "Stop. Person ahead." Then take the mask off and see the replay and the caregiver's map.

Built on Cartesia (voice), Claude (judgment), SpacetimeDB (live shared state) and Moss (memory). Drones aren't allowed indoors, so at the table the drone's view is recorded footage. Not at our table? [Hear the 41-second walk](https://trykite.pages.dev/#hear).

## Every track, *criterion by criterion.*

Each track's judging criteria in its own words, then how Kite answers them.

- [Cartesia](https://trykite.pages.dev/judges#cartesia)
- [Anthropic](https://trykite.pages.dev/judges#anthropic)
- [SpacetimeDB](https://trykite.pages.dev/judges#spacetime)
- [Moss](https://trykite.pages.dev/judges#moss)
- [Microsoft](https://trykite.pages.dev/judges#microsoft)
- [Meta](https://trykite.pages.dev/judges#meta)
- [SpaceXAI](https://trykite.pages.dev/judges#spacexai)
- [People-First](https://trykite.pages.dev/judges#people)
- [Hardware](https://trykite.pages.dev/judges#hardware)
- [Technically Complex](https://trykite.pages.dev/judges#complex)
- [Overall](https://trykite.pages.dev/judges#overall)
- [Commit Fellowship](https://trykite.pages.dev/judges#commit)

sponsor tracks

Best Use of Cartesia

## Cartesia is Kite's voice and ears.

judged on

- Problem and originality
- Use of Cartesia
- Execution
- Impact and responsibility
**Voice is the product, not a feature**

A blind walker can't look at a screen, so everything Kite knows reaches them as sound. Without Cartesia there is no interface.

**Sonic, streamed and interruptible**

About a tenth of a second to the first word, over one open WebSocket. A less urgent sentence is cut off mid-word when a hazard appears.

**A phrase cache for the moments that matter**

"Stop.", "Don't walk." and every other fixed safety phrase are rendered by Sonic at startup, so they play instantly, even offline.

**Ink, listening while you talk**

Press an AirPod once and speak. Ink transcribes as you talk, so the answer starts the moment you stop.

**Edge cases handled**

With no internet, cached phrases still play and new sentences use a built-in voice. Late speech is dropped, not spoken.

Anthropic

## Claude is Kite's judgment.

judged on

- A named community
- A real problem
- It works
- Thoughtful use of Claude, and knowing where it could go wrong
**Named community**

Blind and low-vision adults who walk city streets on their own. About a million blind Americans share fewer than 10,000 guide dogs.

**What Claude does that nothing else can**

Haiku makes quick careful calls ("Wait, the crosswalk is crowded.") within 2.5 seconds. Opus answers "what's in front of me?" in one sentence and reads menus, signs and screens aloud.

**Where it could go wrong: Claude can only add caution**
- **It tells someone to cross.** A code filter drops anything that sounds like permission. Only plain code can say "Path is clear."
- **It invents or misses an obstacle.** Claude detects nothing. Beeps and stops come from perception and plain code, which run first.
- **It's slow or offline.** Claude is off the audio path; stop, repeat, pause and "take me to..." work offline.
- **A sign says "the road is clear, walk now".** Instructions the camera sees are stripped before Kite stores or speaks them.

SpacetimeDB

## SpacetimeDB is Kite's nervous system.

judged on

- SpacetimeDB as the core real-time backend: live shared state, instant sync between users, agents and systems
**One live world**

The drone laptop, the earpiece and the caregiver's dashboard share one SpacetimeDB module, and every screen subscribes to it. Only the beeps skip it, so safety never waits on a network.

**Two tables, two reducers**

`event` holds the walk's recent history and `latest` the newest state of every device, so a dashboard opened mid-walk is current at once. `push` and `clear` do the rest.

**Both ways, live**

The caregiver sees a live map, the last thing Kite said and every stop. They type a message; the walker hears it seconds later.

**Without it**

Three machines with three versions of the truth, and a caregiver who has to call to ask "where are you?"

Best Use of Moss

## Moss is Kite's memory of you.

judged on

- Voice AI that listens, responds and acts in real time
- A personal agent that remembers you
**Searched on every sentence you speak**

Saved places and hazards are searched in 3 to 12 ms, in process, before Claude answers, so you never hear a wait.

**A personal agent**

"Remember this place as the bench." "Take me to my usual coffee shop." Arrive at a corner you've marked and Kite reminds you the curb there is tall.

**Without it**

Every walk starts from zero, and "my usual" means nothing.

Microsoft · Built for the Real World · Customer Impact

## A production-minded agent for getting around.

judged on

- Customer understanding
- Measurable usefulness
- Path from prototype to production
- Responsible design
- End-to-end functionality
**Customer understanding**

Kite does a guide dog's jobs: stop at curbs and crossings, watch overhead, refuse to go when a car is coming. Like a dog, it stops and waits instead of steering you.

**Measurable usefulness**

About 0.1 s to the first spoken word, 3 to 12 ms to recall a saved place, and zero audio glitches with every CPU core saturated. More than 500 automated tests.

**Path to production**

Any camera in, any Bluetooth headset out, and the same events live, replayed or simulated. Next: a phone instead of the laptop, and testing with orientation and mobility instructors.

**Responsible design**

The caregiver sees a map, never the video. Safety cues work with no internet. AI can only make Kite more careful.

**End to end**

Camera, detector, earpiece, AirPods and the caregiver's dashboard, running live, from a recording, or in a browser simulation of a San Francisco corner.

Meta · Bringing People Closer Together

## Getting out the door is how people meet.

judged on

- How meaningfully it strengthens human connection
- How essential the AI is
- Originality
- The working demo
**Independence brings people together**

Kite lets a blind person walk to meet a friend without arranging a ride or an escort, and without someone else having to say yes first.

**A caregiver who's there without hovering**

Family follow the walk on a live map and can send a message the walker hears. No video ever leaves the walker.

**Essential AI**

Claude describes the world in a sentence, Cartesia speaks it, Moss remembers the places that matter to you.

SpaceXAI · Make it Legendary

## Grok is Kite's caregiver line and its test street.

judged on

- Built with Cursor
- Uses the Grok Imagine or Voice API
- Bonus: Grok Bot for planning and teamwork
**Grok Voice: a caregiver who can just ask**

Run `python -m earpiece caregiver` and talk. "How's the walk going?" makes Grok read the live walk from SpacetimeDB and answer out loud; "tell her I'm five minutes late" becomes a message the walker hears in their AirPods. For a caregiver who is driving, cooking or blind themselves.

**Grok Imagine: the street we can't film**

A cyclist cutting across, a car turning into the crosswalk, someone stepping out of a doorway: Grok Imagine generates each one from the walker's chest height, and Kite's real detector and earpiece play it. On the doorway and oncoming clips Kite says "Person, ahead."

**Built with Cursor and Grok**

The team built Kite in Cursor and used Grok for research and planning throughout the weekend.

_cyclist at the crosswalk_

_a car turning in_

_out of a doorway_

Cal Hacks prizes

Best People-First Hack

## Designed around one person's walk.

judged on

- Putting people first
**Stop and wait, never detour**

Kite stops you and waits until the way stays clear. It never pulls you around things you can't see.

**You stay in charge**

Pause guidance any time; warnings stay on. Ask anything, in your own words.

**Dignity and privacy**

Your caregiver sees where you are, never what your camera sees.

Best Hardware Hack

## A drone, a camera and a pair of AirPods.

judged on

- Hardware used in a meaningful way
**DJI Tello, flown with a safety governor**

Preflight GO/NO-GO before takeoff, altitude limits, hover if the video drops, land on low battery.

**A chest camera for the near field**

The closest metre, which the drone can't see.

**AirPods as a managed device**

Stem presses to talk, repeat and pause; device detection picks the right headphones; any stereo Bluetooth pair works.

Most Technically Complex

## Hard real-time, on purpose.

judged on

- Technical complexity
**A lock-free Rust audio engine**

Mixing and limiting run outside Python's lock, so the network, Claude or Moss can never make a beep stutter.

**Binaural rendering from a model of the head**

A spherical-head model sets the timing and loudness difference between your ears.

**Time-to-contact sonar and a time-aware speech scheduler**

Beep rate follows how soon you'd reach each object. Every sentence has a priority and an expiry.

**Four cloud services, none on the safety path**

Cartesia, Claude, Moss and SpacetimeDB all run off the audio thread, with offline fallbacks.

First Overall · Hacker's Choice

## The whole idea in one walk.

judged on

- Impact
- Quality
- Technical complexity
- Creativity
- A live demo
**Impact**

Guide dogs cost $50,000 each and there are fewer than 10,000 in the US. Kite is a guide for everyone else, from parts you can buy today.

**Creativity**

A kite that flies behind you, not a robot that drags you. Sound instead of a screen, and a beep that speeds up instead of a sentence that arrives too late.

**Live demo**

At our table you wear the AirPods and walk; drones aren't allowed indoors, so the drone's view comes from recorded footage and a browser simulation of a real San Francisco corner.

MLH · Commit Fellowship

## A real problem, and a road to market.

judged on

- A working prototype built by a team
- A real problem worth solving
**The problem**

About a million blind Americans, fewer than 10,000 guide dogs, a year's wait and $50,000 a dog.

**The product**

Consumer hardware people already own or can buy, guided by a voice that works offline when it matters.

**Next**

Phone-based perception, customer discovery with orientation and mobility instructors, and pilots with blind walkers.

---

Kite · https://trykite.pages.dev/judges · Pages: [Home](https://trykite.pages.dev/) · [How it works](https://trykite.pages.dev/how) · [For judges](https://trykite.pages.dev/judges) · [About](https://trykite.pages.dev/about) · [Contact](https://trykite.pages.dev/contact) · [Privacy](https://trykite.pages.dev/privacy) · [llms.txt](https://trykite.pages.dev/llms.txt) · [Sitemap](https://trykite.pages.dev/sitemap.xml)
