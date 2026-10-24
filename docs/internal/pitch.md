# Pitch: script, sponsor inserts, Q&A and cue card

Cal Hacks 13, Sun Oct 25, live table judging 10:30 AM to 12:30 PM at the Palace of Fine Arts. Each judge gives us 2 minutes of pitch and 2 of questions. Tables are shared by 2 or 3 teams, so the walk has 3 to 4 m of aisle in a crowded hall. The whole team stays at the table for the full window, or we are not judged.

This follows section 4 of the spec (`docs/internal/spec.md`). The table kit and the failure fixes are in `docs/internal/demo-runbook.md`. Every number here traces to the repo; sources and caveats are in `docs/research/facts.md`. Anything in [TODO] or marked FLAG is not true yet or not confirmed. Do not fill a TODO with a number or a quote we do not have.

## 0. Fill these in before 10:30

1. Presenter: whoever won Saturday night's timed tryout. [TODO: name]
2. There is no interview. We did not talk to a blind traveler this weekend, so the script has no quote and nobody claims user research. Do not add one.
3. Kite-mode collision number from `scripts/sim_metrics.py` (20 walks, seed 1). The blind baseline is 3.05 collisions per walk. The oracle's 0 is not Kite and is never quoted as Kite. [TODO]
4. How the Moss St footage was flown: piloted, follow mode, or handheld (`docs/internal/shoot-day.md`, appendix D). Arthur. [TODO]
5. Whether the footage shows a car turning in from Howard. If it does not, say what it does show. Beat 4 changes.
6. Whether the judge's "Kite, what's in front of me?" works at the table, and how the judge triggers it. The AirPods mic puts the AirPods in mono, so the earpiece uses the laptop mic. The demo driver may have to press talk. Settle this at the 6 PM rehearsal.
7. The key that opens the reveal window and the dashboard's presentation mode (P). The runbook says V for the sim; the live reveal may differ. Check on the day.
8. A written yes from the directors before anyone powers on the Tello at the table. Without it the Tello stays off and nobody says it is on.
9. Kit cost for Q8, from the Target receipts. [TODO]

## 1. Roles at the table

Four people, four jobs. Everyone else at the table stays out of the aisle.

| Role | Does | Default |
|---|---|---|
| Presenter | Speaks every line in section 2. Faces the judge, not the screen. Never touches the laptop. | [TODO: tryout winner] |
| Demo driver | Runs the laptop: starts the walk, triggers the reveal, plays the footage, resets. Watches the earpiece log. Does not speak during the walk except to say "heads up". | Eddy owns the kit |
| Spotter | Walks beside the judge at arm's length for the whole walk, on the side toward the other tables. Steers by touch on the elbow if needed. Never leaves the judge's side, even when the judge is told to stop. | [TODO] |
| Stepper | Waits 2 to 3 m up the aisle, to one side, out of the judge's path. Steps in on the cue, steps out on the cue. Also holds the obstacle for the webcam moment. | [TODO] |

Who answers questions (from the owner table in the spec): Arthur for the Tello and the footage. Rex for perception and the guide dog facts. Karthik for Cartesia and the cached safety phrases, the Claude judgment layer, the camera descriptions ("what's in front of me?" and reading text aloud), stop and wait, walking routes, the SpacetimeDB mirroring, and support for any stereo Bluetooth headphones. Eddy for the dashboard, the sim, the numbers and the kit. The presenter takes the first answer and hands off by name.

## 2. The 2-minute script

Times are the spec's six beats. Lines in quotes are spoken word for word. Lines in brackets are actions.

### Setup before a judge arrives

[Demo driver: earpiece running, `scripts/demo.sh live iPhone`, Do Not Disturb on, headphone volume checked in the hall noise. Sleep mask, headphones and wipes on the table. Tello on the table, powered off. Footage clip open in a player, paused on its first frame. Dashboard and reveal window ready on the laptop.]

[Stepper: standing at the aisle mark, 2 to 3 m from the table, to the side.]

### Beat 1. 0:00 to 0:10. Who it is for, then the judge puts it on

[Presenter holds up the sleep mask and headphones.]

Presenter: "Kite is a guide dog for blind people that flies. That's the drone. Would you try it? Put these on."

[Spotter fits the mask. Demo driver clips the phone to the judge's chest and checks the beeps in their ears. If the judge will not wear someone else's AirPods, hand over the over-ear headphones.]

### Beat 2. 0:10 to 0:50. The judge walks and gets stopped

Presenter: "Walk straight ahead, slowly. My teammate is at your elbow the whole way. Just listen."

[Judge walks. When they are about a meter into the aisle, the stepper walks into their path and stands still.]

[What the judge hears: beeps from the stepper's side that get faster and higher as the gap closes, then at about 1.5 m the voice says "Stop. Person ahead."]

Presenter: "That stop came from code on this laptop, not from a model. Stay where you are."

[Stepper steps out of the path. After the way has stayed clear for 1.5 seconds, the voice says "Path is clear. You can continue."]

Presenter: "It waits until the way stays clear, then it tells you."

Optional, only if the clock reads 0:40 or earlier and the question path works: "Ask it what's in front of you. Say, Kite, what's in front of me?" Claude answers from the newest chest camera frame. If the judge walks on without asking, do not wait for it.

[Spotter takes the judge's elbow and has them stop. Demo driver does not touch anything.]

### Beat 3. 0:50 to 1:10. Mask off, the reveal

[Spotter lifts the mask. Demo driver opens the reveal window and the dashboard.]

Presenter: "Mask off. This is what Kite saw, with faces blurred and nothing saved. There's your stop. The caregiver's screen shows who made each call: Kite's code, not a model."

[Say nothing for a few seconds and let them read the screen. The reveal replays the last 20 seconds with its boxes, distances and the STOP banner. The dashboard shows the walk as a timeline labeled by who acted.]

### Beat 4. 1:10 to 1:35. The drone sees the street

[Demo driver plays the Moss St clip full screen.]

Presenter: "A chest camera sees a few meters. The drone sees the street. Drones are banned in this hall, so this is Tello footage from Moss Street, recorded Saturday at 7 AM, [TODO: flown by hand / follow mode / handheld]. A car turns in from Howard, and Kite says stop. Same detector. The audio is rendered afterward from the detector's events; the walker heard a spotter that day."

[FLAG: that last sentence is required, because on the day the walker did not hear Kite (`docs/internal/shoot-day.md`, appendix C). If the overlay video carries an on-screen caption saying the same, cut the sentence to keep the time. If the clip has no turning car, replace the car sentence with what it does show.]

### Beat 5. 1:35 to 1:50. The number

Presenter: "We measured one thing. In our sim of that corner, a scripted walker with no cues hits something 3.05 times a walk, averaged over 20 walks. With Kite, [TODO: Kite-mode number]. It is a simulation, so it shows the cues work in our world, not that they work for a person."

If the Kite number is not measured by 9:00 AM, cut beat 5 and give the time to beat 4's clip. Do not read the baseline without the Kite number, and never read the oracle's 0.

### Beat 6. 1:50 to 2:00. Prior art and what is real

Presenter: "In 2021, researchers at KIT flew a Tello in front of blindfolded walkers on a leash, and it kept them on the sidewalk. Kite drops the leash: the drone flies behind you, and everything reaches you as sound. Live on this laptop: the voice, Claude and the detector. Recorded: the drone footage. Simulated: the street. Questions?"

[Stop talking after "Questions?". Hand the judge the card with the repo URL and the Devpost link. [TODO: card]]

### Time check

At 2.5 spoken words a second, the presenter's lines run about 200 words, roughly 80 seconds, with the rest taken up by the walk, the reveal and the clip. Beat 6 is the longest spoken beat and may run to about 2:05. That is fine, because questions follow. If the walk runs long, cut the optional question and the "walker heard a spotter" sentence before cutting anything else.

## 3. Staging for the table

The aisle. You have 3 to 4 m and other teams share it.

1. Before judging starts, tell the neighboring teams at the table what we will do: a blindfolded person walks about 4 m and someone steps into their way. Ask them to keep their bags and chairs out of that strip. Be polite; they may be mid-pitch.
2. Mark the start with a piece of tape at the table edge and the stepper's spot with another, 2 to 3 m out and 1 m to the side. Tape is for the floor only, not for any wire.
3. The judge walks away from the table, parallel to it. The spotter is on the side toward the crowd, at arm's length, the whole way.
4. The walk ends when the stop and the "Path is clear" cue have both played, or at 4 m, whichever comes first. The spotter stops the judge before the end of the clear space, not at it.
5. If the aisle is busy at the moment, wait. A short wait is better than a bad walk. If it is busy all window, use the fallback: the judge walks the sim corner with the screen black (runbook, "Running one judge"), guided by the same earpiece.
6. Hall noise can mask the beeps. Check the headphone level in the hall before 10:30, and keep it comfortable. Do not turn it up for a judge who says it is loud enough.
7. The cable. The chest camera is a phone used as a Continuity Camera, so nothing tethers the judge to the table. If the phone path fails, the USB webcam on a 3 m cable is the fallback and the spotter handles the cable.

If the judge does not want to be blindfolded, offer eyes closed. If they still decline, the stepper wears the mask and walks, the judge watches the reveal and holds the obstacle in front of the webcam (runbook, step 6) so they hear the beeps change. Say "that's fine" and keep to the same script.

### Resetting between judges

Do this in the gap, in order. It takes under a minute if the demo driver starts as the judge walks away.

1. Dashboard: press "Clear run" and confirm.
2. Earpiece: leave it running. If it is still speaking: `earpiece/.venv/bin/python -m earpiece send '{"type":"control","action":"stop_speaking"}'`.
3. If the sim was used, reload its browser tab.
4. Wipe the headphones. Swap to the second sleep mask.
5. Rewind the clip to its first frame. Stepper back to the tape.
6. Presenter checks the next judge's badge for a sponsor name (section 5).

Do not run the live walk when a judge is about to leave, or when another team's judge is crossing the aisle. The 60-second version fits those gaps.

## 4. The 60-second version

For a judge who says they are short on time, or who arrives as another team is finishing. Same order, nothing new. The number goes to Q&A if it exists.

| Time | Who | What |
|---|---|---|
| 0:00 to 0:08 | Presenter | "Kite is a guide dog for blind people that flies. Try it for ten seconds. Put these on." Spotter and driver fit the mask and the phone as the presenter speaks. |
| 0:08 to 0:30 | Stepper, spotter | Judge walks. Stepper steps in almost at once. The judge hears the beeps and "Stop. Person ahead." Stepper steps out. "Path is clear. You can continue." Presenter says only: "That stop came from code, not a model." |
| 0:30 to 0:40 | Presenter, driver | Mask off. "That's what Kite saw, faces blurred, nothing saved." Driver shows the reveal. |
| 0:40 to 0:55 | Presenter, driver | Driver plays the last 10 seconds of the Moss St clip, from the car turning in. "A chest camera sees a few meters. The drone sees the street. This is recorded footage, since drones are banned here, and the audio is rendered from the detector afterward." |
| 0:55 to 1:00 | Presenter | "KIT flew a Tello in front of blindfolded walkers on a leash in 2021. We drop the leash. Questions?" |

If the judge asks whether this has been tested with a blind person, use the answer to Q13. Do not skip it.

## 5. Sponsor inserts

When a sponsor judge visits, check the badge and splice the insert in. Each runs about 20 seconds. To make room, drop the optional question in beat 2 and shorten the clip in beat 4 to its car moment. If that is not enough, drop beat 4's disclosure sentence only when a caption on the video covers it.

The criteria below are the ones written in the Cal Hacks 13 hacker guide. We name the sponsor's own words; we claim nothing the repo does not do.

### Cartesia

Guide criteria, 25% each: problem and originality; use of Cartesia (meaningfully integrated into the core experience, voice clear, responsive and natural); execution (main flow works, likely edge cases handled); impact and responsibility (accessibility, privacy and responsible use). The guide also asks for a working demo, a 2-minute demo-first presentation, and a README listing the Cartesia features used and known limitations. It favors a working prototype over a large unfinished idea.

Where it goes: after beat 2, while the judge still has the voice in their head.

Insert: "Voice is the whole interface here, because the walker can't see a screen. Cartesia Sonic streams over one WebSocket, and a warning cuts in mid-sentence. Every fixed safety phrase is pre-rendered in the Cartesia voice and cached, so Stop plays instantly, even offline. Spoken questions go through ink-whisper. The README lists the features and the limits."

Backing: `earpiece/earpiece/speech.py`, `listen.py`, `selftest.py`; README "Cartesia: the voice, both ways".

FLAG: the README says first audio arrives in about 0.1 s in our tests. Re-run `python -m earpiece selftest` on venue Wi-Fi before quoting it. Limits to volunteer if asked: speech input and new sentences need internet (new sentences fall back to macOS `say`), and our plan allows two requests in flight for the whole team, so only one earpiece may warm its cache at a time.

### SpacetimeDB

Guide wording: Spacetime as the core real-time backend, with live shared state or instant sync between users, agents and systems; live dashboards and AI systems coordinating in a persistent world state are named examples; the main thing is that it is meaningfully used, not added on the side.

Where it goes: after beat 3, with the dashboard on screen.

Insert: "Every device on the walk, and the caregiver, shares one live world state. The earpiece writes every event it hears and says into SpacetimeDB. This dashboard reads only from it. The caregiver can type here, and the walker hears it in the headphones. The safety cues stay local on purpose, so a dead network never delays a stop."

Backing: `dashboard/spacetimedb/src/index.ts`, `earpiece/earpiece/spacetime.py`, `dashboard/src/App.tsx`.

FLAG: the caregiver message path was "next" in the spec (section 5.2) and is described as built in the Devpost draft. Do not say the walker hears the caregiver unless it works on the venue Wi-Fi at the rehearsal. If it does, have a teammate type a short message and let the judge hear it. Be ready for "so it's not load-bearing for safety": that is correct, and is a design choice (Q10 and Q11).

### Anthropic

Guide criteria: a named community (not "students"), a real problem you lived or talked to someone about, it works, and thoughtful use of Claude, including knowing where it could go wrong for the people using it.

Where it goes: replaces the second half of beat 5, or follows it.

Insert: "Our community is blind pedestrians in San Francisco, about 20,000 people by the census who are blind or have serious trouble seeing. We did not get to talk to one this weekend, so we built from published research. Claude does three jobs: one careful sentence at a hazard, a description of what the camera sees, and answers to questions. Where it can go wrong is saying a street is safe or inventing an obstacle. So code checks every sentence, Claude can only make Kite more careful, and only code can end a stop."

Backing: `earpiece/earpiece/judgment.py`, `vision.py`, `brain.py`, `test_judgment.py`; README "where Claude could go wrong"; the 20,000 figure is ACS 2024 for San Francisco County (20,221 with vision difficulty, margin of error about 2,600), `docs/research/facts.md`.

FLAG: "20,000" counts people with serious trouble seeing, which is wider than blind. Say it as written. The criterion asks for someone we talked to and we have no one, so we are weaker there. Say so rather than imply otherwise; Karthik takes the Claude questions.

### Microsoft

Guide criteria for "Built for the Real World" and the Customer Impact Award: customer understanding, measurable usefulness, path from prototype to production, responsible design, end to end functionality.

Where it goes: replaces beat 5 and the second sentence of beat 6.

Insert: "Customer: we built from guide dog school figures and the KIT user study, and we have not yet tested with a blind walker. Usefulness: [TODO: Kite-mode collision number against the 3.05 baseline]. Production: every piece sits behind a seam, so a phone replaces the laptop by changing a setting; that is our plan, not something built. Responsible: no stored video, faces blurred, and the safety path stays on this laptop. End to end: you just walked it."

Backing: README "Microsoft: built for the real world", `scripts/sim_metrics.py`, `earpiece/tests/test_app_stress.py`, `SECURITY.md`.

FLAG: customer understanding and usefulness are our weakest criteria: no interview, and no Kite-mode number until it is measured. Without it, say "usefulness: we have a baseline and have not yet measured Kite", which is honest and still shows we measure.

## 6. Judge Q&A: the 15 hardest questions

Short answers, grounded in the repo. FLAG marks an answer that depends on something not yet true. Keep each answer to about 20 seconds, then stop. If we do not know, say "we don't know yet" and what we would do to find out.

### Q1. What is real and what is simulated?

Real and live on this laptop: the earpiece, the voice, Claude, the detector on the phone chest camera, SpacetimeDB and the dashboard. Real but recorded: the Tello footage from Moss St, taken at 7 AM Saturday. Simulated: the street in the sim, which is Moss and Howard built from OpenStreetMap, not a 3D scan, and the collision number. The drone never flew here, and Kite has never been tested with a blind walker.

FLAG: the detector on the phone is the stand-in (YOLO11n plus depth) unless Rex's `streetseg` landed. Name the one that is running. "Confirmed on the phone camera" is a TODO in the README until the rehearsal.

### Q2. Isn't this Flying Guide Dog?

It is the closest prior work and we cite it. In 2021 a KIT team flew a Tello in front of blindfolded walkers, found the sidewalk with segmentation, and the walker followed a string tied to it. Their study had six sighted, blindfolded people and a questionnaire, and it kept them centered and was easy to learn; traffic lights still needed work. Kite drops the leash and flies behind the walker, adds a chest camera for the ground at their feet, and delivers everything as sound, with stop rules taken from how guide dogs behave. As far as we found, nobody has put those together. We do not claim to beat their results, because we have not run a comparison.

### Q3. What if it misses a car?

It can, and we have not measured how often. The stand-in detector only knows common object classes, cannot tell a parked car from a moving one, and in the sim a car turning in from Howard behind and to the right of a walker facing the zebra only enters the camera's view about 3 m away. That is why Kite never says a street is safe to cross, and why the cane stays. Silence from Kite is never permission. Kite is an extra source of warnings, not a replacement for judgment.

FLAG: the README reports that detector recall drops a lot on sim frames with Google 3D Tiles (people about 9 to 10% against about 60% on the OpenStreetMap world). Quote it only if asked, from `sim/README.md`, and say the live walk does not use the tiles.

### Q4. Why behind the walker and not ahead?

A drone ahead of a walker is between them and the street, and it has to hold a leash to be useful. Behind and above, it sees the route over the walker's head and stays out of their way. The cost is real: from 30 cm above and 1 m behind, the fixed camera cannot see the ground for the first few meters, which is the job of the chest camera, and a car coming in from the side shows up late. We chose it as a design call and have not tested it against a drone in front.

### Q5. What are the Tello's speed and battery limits?

The standard Tello tops out near 0.96 m/s at full stick (extrapolated from a stick-30 test), and people walk at 1.2 to 1.3 m/s. We capped it at about 0.5 m/s, so the walker in our footage walks slowly. One battery gives about 6 to 8 minutes with video on. It has no GPS, it lands itself 15 seconds after it stops hearing us, and the KIT team reported it is too light to hold steady in wind. So the drone is a prototype of the view, not a product. The live walk here uses the chest camera for exactly that reason.

### Q6. Why not just a cane or a guide dog?

Use both if you can. A guide dog is the standard. Guiding Eyes puts the cost of raising and training one at $50,000, paid by donors, not by the blind person, and at Guide Dogs for the Blind the wait after applying was 7 to 10 months in 2022. Fewer than 10,000 guide dogs work in the US against about 1 million Americans who are blind. In a 2022 survey of 236 non-users, 31% did not want the responsibility of a dog and 22% could not afford one. A cane finds what is a step or so ahead, and the federal access rules treat anything between 27 and 80 inches high as a hazard to a cane user. Kite is meant to sit beside the cane, with the walker's hands free for it.

FLAG: say "Guiding Eyes puts the cost at $50,000", not "a guide dog costs $50,000", and never say a blind person pays it. Rex signs off on these figures; sources and caveats are in `docs/research/facts.md`. We have not shown that Kite does any of this better than a cane.

### Q7. What about the privacy of the cameras?

The caregiver dashboard shows positions and events, never video. Frames we send to Claude to answer "what's in front of me" are not mirrored anywhere. The reveal window pixelates faces and writes nothing to disk, and we store no frames during the live demo. The detector runs on the laptop, and the safety cues stay on localhost.

FLAG: the honest gaps. A frame does go to Claude's API when the walker asks a question. The raw Tello takes from Moss St were recorded on a public street and sit on team laptops; only the overlay video has heads blurred. Bystanders near a walker have not agreed to be seen by a chest camera, and we have no answer for that beyond not storing frames. Say so.

### Q8. What would it cost?

We do not have a production price. The prototype is consumer parts: a Tello, a phone as the chest camera, headphones and a laptop, and the Cartesia and Claude calls cost money per use. [TODO: kit cost from the receipts and the API cost per walk, if Eddy measured it.] Comparing with a $50,000 guide dog is not fair, since that figure is what a school spends to raise and train one dog over a working life of about six to ten years, and Kite is not a trained animal. We would rather compare the cost per walker once it works on a phone.

FLAG: no cost number exists in the repo. Do not make one up. If asked "so is it cheaper", the honest answer is "that is the goal, and we have not shown it."

### Q9. Who did you talk to?

"We didn't get to talk to a blind traveler this weekend. We built from published research: guide dog schools' own figures and the KIT Flying Guide Dog user study of blindfolded walkers. Testing with orientation and mobility instructors is our next step."

FLAG: the Anthropic criterion (someone you talked to) and the Microsoft criterion (customer understanding) both lean on this answer, and we have nothing to offer. Say it plainly and do not hedge.

### Q10. How does Claude not tell someone to cross?

It is not given the chance to. Stops, sonar beeps and "Don't walk" are decided in deterministic code before Claude sees anything. Claude answers through one strict tool that returns at most one short sentence and an optional upgrade of a hazard already on the list. Code drops any sentence that sounds like permission ("safe to cross", "go ahead", "keep walking", "detour"), anything over 120 characters, any move to critical, any downgrade, and any answer slower than 2.5 seconds. Only the deterministic code can end a stop, and nothing in Kite says a street is safe.

FLAG: the permission filter is a phrase list with tests (`test_judgment.py`). A paraphrase we did not think of could get past it. The stronger protection is that Claude cannot clear a stop. Claude's answer to "what's in front of me?" can also be wrong, because it describes a frame. Say both if pressed. Karthik built the judgment layer and the camera descriptions; hand this one to him.

### Q11. What happens offline?

The safety path never needed the internet. Sonar beeps, stops, "Don't walk" and the cached safety phrases run on this laptop. Local voice commands (stop, repeat, pause, louder) still work. With Claude down, open questions get a short offline answer. With Cartesia down, new sentences fall back to macOS `say` and voice questions stop working. With SpacetimeDB down, we start a local server on the same laptop (`KITE_OFFLINE=1 scripts/demo.sh sim`) and the dashboard keeps working.

FLAG: the spec says these fallbacks are tested by Saturday's freeze. Karthik owns the cached safety phrases and the stop and wait logic; Eddy owns the offline script. Confirm at the rehearsal that you have actually run `scripts/offline.sh` and the demo in airplane mode.

### Q12. Are you allowed to use Google imagery?

For this one street, yes. Google's Maps Platform policy lists object detection as a prohibited use, so we asked. Google's representatives at Cal Hacks told Eddy on Saturday at about 3 AM that we could use Photorealistic 3D Tiles and Street View for this hackathon, including running our detector on rendered frames, only for Moss St between Folsom and Howard. The sim masks and counts tile requests outside that block, we store no Google imagery, and their attribution stays on screen. If the permission did not apply, the sim uses the OpenStreetMap world instead.

FLAG: `docs/google-permission.md` still has TODOs for the representative's name and a written confirmation. As it stands the permission is verbal. Say "verbal" if asked. Fill it in, or say we hold it verbally and fall back to OpenStreetMap if the judge wants us to.

### Q13. Would a blind person wear this?

We do not know. No blind walker has tried it. What worries us, and what we would test first: earphones that cover the ears take away the street sound a blind walker relies on, a drone's noise behind the head could mask the very sounds that matter, and a laptop on the back is not something to wear daily. The design choices we can defend are that Kite stops and waits instead of steering, the walker stays in charge, and critical warnings cannot be muted by any remote command.

FLAG: nothing in the repo tests ear coverage or drone noise. This is the answer with the most risk of sounding naive if we pretend otherwise. If a blind judge or an O&M instructor asks, listen first and write down what they say.

### Q14. What is next?

In order: test with orientation and mobility instructors and blind walkers outdoors, with a spotter. Then fly follow mode outdoors behind and above the walker under the safety governor already in the drone layer; it has only run in a closed-loop simulation and has never flown. Then read pedestrian signals from closer range or from the chest camera, since a signal 20 m away is about 14 pixels tall in the Tello stream. After that, put perception on a phone so the laptop goes away.

### Q15. Why a drone at all?

Honestly, at this table the drone is off, and the live part is the chest camera. The drone view is shown in recorded footage and in the sim. What it adds is a camera that is not on the walker's body and looks along the route, so it can see crowds, a crossing and a car turning in before the walker reaches the curb. But it only sits about 30 cm above head height, so it does not see much farther than a camera on a hat would. We have not shown that a flying camera beats a worn one. If a judge presses, say so and describe what we would measure: warning distance on the same route from each camera.

FLAG: this is the weakest answer. The team should agree on one line before 10:30. Do not claim the drone sees far past the walker's head.

## 7. Cue card

Print this page on its own. One page, large type.

<div style="page-break-after: always"></div>

# KITE CUE CARD

Roles: Presenter [__]. Driver [__]. Spotter [__]. Stepper [__].
Spotter at arm's length, always. Nobody leaves the table.

## 2:00 SCRIPT

0:00  "Kite is a guide dog for blind people that flies. That's the drone. Put these on." Mask, headphones, phone on chest.

0:10  "Walk straight ahead. Just listen." Stepper steps in. Kite: "Stop. Person ahead." Say: "That stop came from code, not a model." Stepper steps out. Kite: "Path is clear." Optional: "Kite, what's in front of me?"

0:50  Mask off. Reveal. "This is what Kite saw. Faces blurred, nothing saved. The caregiver sees who made each call."

1:10  Moss St clip. "A chest camera sees a few meters. The drone sees the street. Recorded, flown [__]. A car turns in. Same detector. Audio rendered afterward; the walker heard a spotter."

1:35  Sim: 3.05 collisions a walk with no cues, [__] with Kite. No Kite number: cut the beat.

1:50  "KIT flew a Tello in front of blindfolded walkers on a leash in 2021. Kite drops the leash: the drone flies behind, everything reaches you as sound. Live: voice, Claude, detector. Recorded: drone footage. Simulated: the street. Questions?"

## 60 SECONDS

Mask on, walk, stop, clear (30 s). Mask off, reveal (10 s). Last 10 s of the clip (15 s). KIT line, questions (5 s).

## SPONSOR (20 s)

Cartesia: voice is the interface; cached safety phrases; cuts in mid-sentence; README has limits.
SpacetimeDB: one live state; dashboard reads only from it; caregiver message comes back; safety stays local.
Anthropic: blind pedestrians in SF, no interview; three Claude jobs; code filters it; only code ends a stop.
Microsoft: customer, usefulness, production, responsible, end to end. Say which parts are plan.

## NEVER SAY

"Safe to cross." The oracle's 0 as Kite's. A guide dog "costs a blind person $50,000". The drone flew here. That we talked to or tested with a blind person. "Faster" about Flying Guide Dog.

## FAILS

Mic or phone camera dies: sim walk, screen black (B), same earpiece. Cartesia or Claude down: nothing to do, safety is local. Internet gone: `KITE_OFFLINE=1 scripts/demo.sh sim`. AirPods drop: laptop speaker, re-pair. Anything else: backup video, talk over it.

## ANSWERS FIRST WORD

Real vs sim: live, recorded, simulated. Misses a car: it can; cane stays; never says safe. Behind: out of the way; chest cam covers the ground. Tello: 0.96 m/s max, 6 to 8 min. Who we talked to: nobody; published research. Claude: only code can end a stop. Google: Moss St only, verbal. Cost: no number yet. Drone: weakest, agree a line.
