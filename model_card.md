# 🎧 Model Card: Music Recommender Simulation

## 1. Model Name

**VibeFinder 1.0**

It finds songs that match the "vibe" you ask for.

---

## 2. Goal / Task

VibeFinder suggests songs you might like.

You tell it three things: a genre, a mood, and how much energy you want.

It looks at every song in the catalog and picks the five that fit you best.

It is built for classroom exploration, not for real users yet.

The goal is to be easy to explain. You can always see *why* a song was picked.

---

## 3. Data Used

The catalog has **20 songs**.

Each song has these features:

- Title and artist (just labels)
- Genre (like pop, rock, lofi)
- Mood (like happy, chill, intense)
- Energy, a number from 0.0 (calm) to 1.0 (hype)
- A few extra numbers (tempo, valence, danceability, acousticness) that we do **not** use yet

Limits of the data:

- It is tiny. Only 20 songs.
- There are 17 different genres, so most genres have just one song.
- The songs lean high-energy. The average energy is about 0.61.
- Many real tastes and genres are simply missing.

---

## 4. Algorithm Summary

Every song starts with a score of 0. We add points for each thing it gets right.

- **Genre match:** if the song's genre is the one you asked for, it gets **+2.0**.
- **Mood match:** if the song's mood matches, it gets **+1.0**.
- **Energy fit:** the closer the song's energy is to your target, the more points it earns, up to **+1.5**.

Then we sort all songs from highest score to lowest and show you the top five.

Genre is worth the most because it is the steadiest sign of taste. Mood is worth less because it changes with your day. Energy is a sliding scale, not pass-or-fail, so it acts like a tie-breaker.

I also ran an experiment where I doubled energy and halved genre. It changed the scores but barely changed the rankings.

---

## 5. Observed Behavior / Biases

The clearest weakness I found is an **energy-gap bias that quietly favors mid- and high-energy listeners**. The catalog itself leans loud — the average song energy is about 0.61, and 9 of the 20 songs sit at 0.7 or above while only 5 fall below 0.4 — so a low-energy user simply has fewer close matches to draw from. Because the energy score is based on how close a song is to your target, users with *extreme* tastes (like a target of 0.05 or 0.98) find almost every song "far away," so their scores get squished and their top-5 becomes a "least-bad match" instead of a great fit. Meanwhile a user around 0.5 gets sharp, confident results. On top of that, genre is an exact word-match across 17 nearly-unique genres, so a close tag like "indie pop" earns zero for a "pop" fan — a filter bubble that reinforces the exact label and never surfaces cross-genre discovery. In short, the system serves the "average" mid-to-high-energy, mainstream listener well and treats everyone at the edges as an afterthought.

---

## 6. Evaluation Process

**Profiles I tested.** I ran three very different listeners: **High-Energy Pop** (upbeat pop, energy 0.9), **Chill Lofi** (calm lofi, energy 0.25), and **Deep Intense Rock** (hard rock, energy 0.85). I picked these because they pull in different directions, so each should get a clearly different top-5.

**What I looked for.** For each profile I checked if the #1 pick was an obvious "yes," and if the songs below it still felt related to the request.

**What surprised me.** The same high-energy songs kept showing up for different users. And when I doubled the energy weight, the *order* of the top-5 barely changed — only the scores did. The system leans hard on energy.

**Why "Gym Hero" keeps showing up for the "Happy Pop" listener.** Gym Hero is a pop song with very high energy, but its mood is *intense*, not *happy*. The Happy Pop listener wants pop + happy + high energy. Gym Hero nails two of the three: it is pop and it is high energy. We give points for each thing a song gets right, so two-out-of-three is still a strong score. It can't beat "Sunrise City" (which is all three), but it beats everything else. So Gym Hero lands at #2 — being loud and pop is enough, and the missing "happy" only costs it one point.

**Comparing the profiles, two at a time.**

- **High-Energy Pop vs. Chill Lofi.** Near opposites. Pop's list is all loud, bright songs near energy 0.8–0.9. Lofi's list is all quiet, mellow songs near 0.35–0.42. The energy target does most of the work, so calm and hype listeners pull completely different corners of the catalog.

- **High-Energy Pop vs. Deep Intense Rock.** These overlap. Both want high energy, so loud songs like Gym Hero appear in both lists. Genre and mood break the tie: Pop's #1 is "Sunrise City" (happy pop), Rock's #1 is "Storm Runner" (intense rock). The shared energy craving pulls a shared pool of loud songs, then genre/mood decides each winner.

- **Chill Lofi vs. Deep Intense Rock.** The most extreme split. Lofi wants 0.25, Rock wants 0.85, and almost nothing overlaps. Lofi gets soft instrumental tracks; Rock gets the loudest songs in the catalog. This is the clearest proof that the energy dial really steers the results.

No numeric accuracy scores were computed. I evaluated by reading the top-5 for each profile and comparing them by hand.

---

## 7. Intended Use and Non-Intended Use

**Intended use:**

- A classroom demo to show how scoring turns data into recommendations.
- A safe sandbox to test how weights change results.
- Learning about bias in recommender systems.

**Not intended for:**

- Real users or a real music app.
- Any decision that matters (it is a toy with 20 songs).
- Judging an artist, genre, or person's taste as "good" or "bad."
- Any case where fairness across all listeners is required.

---

## 8. Ideas for Improvement

1. **Fuzzy genre matching.** Give partial credit for close genres, so "indie pop" counts for a "pop" fan. This would pop the filter bubble a little.
2. **Use the extra features.** Tempo, valence, and danceability already sit in the data. Adding them would break ties more fairly than energy alone.
3. **Add variety to the top-5.** Right now the list can be five near-identical songs. Mixing in one or two surprise picks would help discovery.

---

## 9. Personal Reflection

**My biggest learning moment.** It clicked for me that a recommender is really just a scoring rule plus a sort. Nothing magic. But the *small* choices — like whether a genre match is worth 2.0 or 0.5 — quietly decide who the system serves well. My biggest "aha" was the weight experiment: I doubled the energy weight expecting the recommendations to change a lot, and the rankings barely moved. That taught me that tuning a knob is not the same as fixing a problem. The real issue was deeper, in the data.

**How AI tools helped, and when I double-checked them.** Using an AI coding assistant sped me up a lot. It applied the weight changes across the file quickly, explained the score math term-by-term, and helped me spot the energy-gap bias by looking at the data spread. But I did not take its word for anything. When it told me Storm Runner would still rank #1 at a lower genre weight, I actually ran the code to confirm the numbers. When it summarized the genre and energy distribution, I re-ran a quick count myself. The pattern I settled on: let AI draft and explain, but run the real code before I believe any claim about results.

**What surprised me about simple algorithms.** I was surprised that adding three plain numbers together can *feel* like a real recommendation. There is no machine learning here, no crowd of users, just "add points for what matches." Yet the top picks often felt right, like the system "got" the user. That was a little unsettling too, because it means something that feels smart can still be shallow and biased underneath.

**Bias hides in the data, not just the code.** The catalog leaned high-energy, so calm listeners got worse results no matter how I tuned the weights. Real music apps can favor certain tastes just from what songs they happen to have. Now I think a lot more about *who gets left out* when I use a recommendation app.

**What I would try next.** If I kept going, I would add fuzzy genre matching so close tags like "indie pop" count for a "pop" fan, use the extra features (tempo, valence, danceability) that are already sitting unused in the data, and add a little variety to the top-5 so it is not five near-identical songs. I would also balance the catalog so calm listeners get as fair a shot as hype listeners.
