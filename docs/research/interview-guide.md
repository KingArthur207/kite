# Kite: interview guide

For a 10-minute call with a blind or low-vision person, or with an orientation and mobility (O&M) instructor. Rex runs the interview. Anyone on the team can run it if Rex isn't free, as long as they follow the consent part word for word.

The aim is to learn what we got wrong, and to get one or two sentences we may quote with permission. It is not a sales call. If they say Kite is a bad idea, that is a useful answer and it goes in the notes as said.

## Before the call

- One person asks, one person takes notes. Notes are typed, not recorded, unless they say yes to recording.
- Have the one-line description ready (below). Don't demo or share a screen unless they ask; many people on the call won't be able to see it.
- Keep to 10 minutes. Set a timer and tell them when 8 minutes have passed.

## Opening and consent (1 minute)

Say roughly this:

> Thanks for taking the time. I'm [name], a student on a team of four at Cal Hacks, a hackathon in San Francisco this weekend. We're building a prototype called Kite for blind pedestrians: a small drone flies just above and behind the walker, a camera on their chest watches the ground, and they hear beeps and short spoken warnings in headphones. It's a weekend prototype and hasn't been tested with blind walkers.
>
> I have about eight questions and will stop at ten minutes. You can skip any question or end the call whenever you like.
>
> Is it OK if I take notes? Would you be OK with recording, only so we can quote you accurately? We'd delete it after the weekend.
>
> We might want to quote a sentence or two in our project writeup and our two-minute pitch to judges. If we do, how would you like to be named: full name, first name, a description like "an O&M instructor in San Francisco", or not at all? We'll read you the exact quote before we use it, and if you say no then, we drop it.
>
> If you work for an organisation, are you speaking for yourself or for them?

Write down their answers to all four: notes, recording, how to be named, own view or the organisation's. No quote goes into `docs/devpost.md` or the pitch without the second yes, after they hear the exact words.

## Questions (about 8 minutes)

Pick the ones that fit the person. For an O&M instructor, ask about their students instead of about themselves. Ask follow-ups like "can you tell me about a time that happened?" rather than offering options.

1. How do you get around on a usual day, for example walking to somewhere you haven't been before? Which tools do you use (cane, dog, apps, asking people)?
2. What does your cane or dog not tell you that you'd want to know? When does that matter most?
3. Tell me about crossing a street. What makes a crossing easy or hard? What do you listen for?
4. What happens in crowds, or around things that change, like construction, scooters, cars pulling across the sidewalk or something at head height?
5. When an app or device talks to you while you walk, what makes you trust it? What makes you stop listening?
6. We use beeps that come from the direction of an obstacle and speed up as it gets closer, plus short spoken warnings like "Stop. Person ahead." Does that sound useful or tiring? Would it cover sounds you need to hear?
7. Kite never says a street is safe to cross. It only says stop, or wait, or describes what it sees. Does that seem right to you? What should it say at a crossing?
8. A drone above you and a camera on your chest record the street around you. How do you feel about that, for you and for the people walking past? What would you want done with the video?
9. Kite can show a family member or caregiver where you are and what warnings you got, without any video. Would you want that? Who decides when it's on?
10. What would make you stop using something like this after a week?

Last minute: "Is there anything we should have asked and didn't? Is there anyone else you think we should talk to?" Then thank them, and say when they'll hear the quote back if they agreed to one.

## What we must not claim

To the person on the call, and later in anything we write about the call:

- That Kite is safe, tested or ready to use. It is a weekend prototype and no blind walker has used it.
- That Kite tells anyone when it is safe to cross. It doesn't, and that's a design rule.
- That Kite replaces a cane, a guide dog or O&M training. At most it is something used alongside them.
- That the person or their organisation endorses Kite, unless they say so in those words and agree to be quoted saying it.
- That our sim collision numbers show Kite makes blind people safer. The sim uses scripted walkers, not people.
- That the drone footage was autonomous, unless it was. Say whether a teammate piloted it.
- That drones can legally follow a pedestrian down a city street. We haven't checked FAA and San Francisco rules for that, and the venue bans drones.
- The "2% of blind people use a guide dog" figure. See `docs/research/facts.md`: it dates from around 1990 and nobody has a current number.
- Anything the person said that they asked us to keep off the record.

When writing the notes up, use their words, mark anything paraphrased as a paraphrase, and don't tidy a quote beyond removing "um".

## Who to contact

Public contact pages only. Check each page before writing; details change. Nothing has been sent to anyone yet.

| Organisation | Why them | Contact |
|---|---|---|
| LightHouse for the Blind and Visually Impaired, San Francisco | The main SF agency for blind and low-vision adults, with O&M instructors on staff. It's in SoMa, close to the Moss St corner we modeled. | [lighthouse-sf.org/contact-us](https://lighthouse-sf.org/contact-us/), info@lighthouse-sf.org, 415-431-1481, 1155 Market St, 10th floor |
| Guide Dogs for the Blind, San Rafael | Guide dog school with O&M staff, and the source of several figures in `facts.md`. Could correct us on what guide dogs actually do. | [guidedogs.com/contact-us](https://www.guidedogs.com/contact-us), 350 Los Ranchitos Rd, San Rafael |
| National Federation of the Blind of California, San Francisco chapter | Blind people speaking for themselves, not as service providers. Expect pointed views on tech built for blind people without them. | [nfbcal.org](https://nfbcal.org/). The SF chapter has met at LightHouse, so LightHouse may also pass a message on. |
| UC Berkeley Disabled Students' Program | Can forward a request to blind and low-vision students who might want to talk. They won't share names. | [dsp.berkeley.edu/contact-us](https://dsp.berkeley.edu/contact-us), dsp@berkeley.edu, 510-642-0518 |
| Center for Accessible Technology, Berkeley | Assistive technology specialists, at the Ed Roberts Campus. | [c4at.org](https://c4at.org/about_us.html) |
| San Francisco State University O&M Program | One of the oldest O&M training programs in the US (founded 1966). Faculty and students are O&M specialists. | [mobility.sfsu.edu](https://mobility.sfsu.edu) |
| Orientation Center for the Blind, Albany (California Department of Rehabilitation) | Residential training center with O&M instructors, ten minutes from Berkeley. A state agency, so a weekend reply is unlikely. | [dor.ca.gov](https://www.dor.ca.gov/), 400 Adams St, Albany |
| Smith-Kettlewell Eye Research Institute, San Francisco | Researchers who build and study navigation aids for blind people. Good for a prior-art sanity check, less for a user quote. | [ski.org](https://www.ski.org/) |

Saturday is a hard day to reach anyone at an agency. The best odds are individuals: someone a teammate already knows, a Cal student through DSP, or an NFB member. Send the DM version where a personal contact exists.

## Outreach drafts

Drafts only. Don't send without Rex or Eddy reading them first, and fill in the brackets.

### Email

Subject: 10-minute call today? Student project on navigation for blind pedestrians

> Hi [name],
>
> I'm [name], a [Berkeley] student on a team of four at Cal Hacks in San Francisco this weekend. We're building a prototype for blind pedestrians: a small drone flies just behind and above the walker, a chest camera watches the ground, and the walker hears directional beeps and short spoken warnings in headphones.
>
> We'd like 10 minutes with a blind or low-vision traveler or an O&M instructor before we go further, to hear what we've misunderstood. Would you, or someone you could pass this to, have time for a short phone or video call today (Saturday) between [time] and [time]?
>
> It's a student project and isn't for sale. We won't quote anyone without their permission, and we'll read back any quote before using it.
>
> Thank you,
> [name]
> [phone]

### Direct message

> Hi [name], I'm on a student team at Cal Hacks this weekend building a navigation prototype for blind pedestrians (a small drone behind the walker, plus audio cues). Could we have 10 minutes of your time today, by phone or video, to hear what we're getting wrong? No quotes without your OK. Any time between [time] and [time] works.
