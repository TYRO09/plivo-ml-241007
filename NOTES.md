# NOTES

The model scores each pause using only audio before it: terminal prosody (final
pitch relative to the speaker's own running median *and* to their earlier
pauses in the same turn, declination residual, fall fraction), final-syllable
elongation measured as spectral flux (a spectrally "frozen" vowel means the
speaker is holding the floor), energy decay into the pause, speaking-rate
slowdown, channel-normalised cepstra of the last 200 ms, and turn-so-far
context (pause index, speech time so far, previous pause statistics). The most
useful family is the **within-turn relative** one — comparing this pause to the
earlier pauses of the same turn cancels speaker, handset and language variation,
and it is the strongest single feature in Hindi (AUC 0.77 against long holds).
Because the scorer only penalises a hold when the action delay is shorter than
that hold, I weight training samples by a sigmoid in pause duration, which
reframes the objective as "EOT vs *long* hold" instead of "EOT vs any pause".

It still fails on list-like turns where the speaker pauses after each item with
fully final prosody ("…two pizzas, …one garlic bread, …"); those produce most of
the false cutoffs. It also fails when a turn ends on a rising or level contour
(confirmation questions), and on first pauses with under a second of speech
context. Held out over 8 fold shuffles the model reaches 1184 ± 49 ms on English
(baseline 1600, ~8σ) but only 835 ± 16 ms on Hindi (baseline 850) — a gain of
0.9σ, with 38% of shuffles no better than the baseline, so **I cannot claim a
reliable Hindi improvement from 100 turns**. The reason is structural: 83% of
Hindi holds here are ≤0.5 s, so a plain 850 ms timer is already near the best a
threshold policy can do, whereas 63% of English holds exceed 0.5 s and leave
real headroom. Since the hidden set is mostly Hindi, expect roughly baseline
there.

I reimplemented the Voice Activity Projection objective (Ekstedt & Skantze
2022) from scratch and it did **not** beat supervised prosody out of fold,
because 36 minutes of single-channel audio yields only ~1,600 distinct silence
events and most of that paper's signal comes from the second speaker's channel,
which this data does not have. With one more day I would do semi-supervised
training on those ~1,600 unannotated VAD-detected silences using the full
pause-level feature set and self-supervised "does the user resume within 2 s"
targets — the right way to get VAP's data multiplier with far better features
than my frame-level ones — plus speed/gain/noise augmentation, a sequence model
over the pauses within a turn, and an hour spent listening to the 20 worst
false cutoffs.
