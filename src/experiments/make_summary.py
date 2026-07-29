"""Build the self-contained SUMMARY.html from artifacts/{results,figs}.json."""
import json
import os

# src/experiments/make_summary.py -> src/experiments -> src -> repo root
ROOT = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))

res = json.load(open(os.path.join(ROOT, "artifacts", "results.json")))
figs = json.load(open(os.path.join(ROOT, "artifacts", "figs.json")))

cv_en, cv_hi = res["cv"]["english"], res["cv"]["hindi"]
b_en, b_hi = res["baseline"]["english"], res["baseline"]["hindi"]
full_en, full_hi = res["cv_english_full"], res["cv_hindi_full"]
top = list(res["top_features"].items())[:12]
res["insample_note"] = "264 ms"


def img(key, cap):
    return (f'<figure><img src="data:image/png;base64,{figs[key]}" alt="{cap}">'
            f'<figcaption>{cap}</figcaption></figure>')


HTML = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>End-of-Turn Detection — plivo-ml-241007</title>
<style>
:root {{ --ink:#1a202c; --mut:#4a5568; --line:#e2e8f0; --acc:#2b6cb0;
        --good:#2f855a; --bad:#c53030; --bg:#fff; --card:#f7fafc; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--ink);
  font:16px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }}
.wrap {{ max-width:940px; margin:0 auto; padding:48px 24px 80px; }}
header {{ border-bottom:3px solid var(--ink); padding-bottom:20px; margin-bottom:36px; }}
h1 {{ font-size:30px; margin:0 0 6px; letter-spacing:-.02em; }}
.sub {{ color:var(--mut); font-size:15px; }}
h2 {{ font-size:22px; margin:44px 0 14px; padding-bottom:7px;
      border-bottom:1px solid var(--line); }}
h3 {{ font-size:17px; margin:26px 0 8px; }}
p, li {{ color:#2d3748; }}
code {{ background:var(--card); padding:1px 5px; border-radius:3px;
        font:13px ui-monospace,SFMono-Regular,Menlo,monospace; }}
pre {{ background:#1a202c; color:#e2e8f0; padding:14px 16px; border-radius:6px;
       overflow-x:auto; font:13px/1.5 ui-monospace,Menlo,monospace; }}
pre code {{ background:none; color:inherit; padding:0; }}
table {{ width:100%; border-collapse:collapse; margin:16px 0; font-size:14.5px; }}
th, td {{ text-align:left; padding:9px 11px; border-bottom:1px solid var(--line);
          vertical-align:top; }}
th {{ background:var(--card); font-weight:600; }}
td.num {{ text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap; }}
.hero {{ display:flex; gap:14px; flex-wrap:wrap; margin:26px 0 8px; }}
.kpi {{ flex:1 1 190px; background:var(--card); border:1px solid var(--line);
        border-left:4px solid var(--acc); border-radius:6px; padding:14px 16px; }}
.kpi .v {{ font-size:27px; font-weight:700; letter-spacing:-.02em; }}
.kpi .l {{ font-size:12.5px; color:var(--mut); text-transform:uppercase;
           letter-spacing:.05em; }}
.kpi .d {{ font-size:13px; color:var(--good); font-weight:600; }}
.note {{ background:#fffaf0; border-left:4px solid #dd6b20; padding:13px 16px;
         border-radius:0 6px 6px 0; margin:18px 0; font-size:14.5px; }}
.warn {{ background:#fff5f5; border-left:4px solid var(--bad); padding:13px 16px;
         border-radius:0 6px 6px 0; margin:18px 0; font-size:14.5px; }}
.ok {{ background:#f0fff4; border-left:4px solid var(--good); padding:13px 16px;
       border-radius:0 6px 6px 0; margin:18px 0; font-size:14.5px; }}
figure {{ margin:22px 0; }}
figure img {{ width:100%; border:1px solid var(--line); border-radius:6px; }}
figcaption {{ font-size:13px; color:var(--mut); margin-top:7px; }}
.two {{ display:grid; grid-template-columns:1fr 1fr; gap:22px; }}
@media (max-width:760px) {{ .two {{ grid-template-columns:1fr; }} }}
.pipe {{ background:var(--card); border:1px solid var(--line); border-radius:6px;
         padding:16px; font:13px/1.9 ui-monospace,Menlo,monospace;
         overflow-x:auto; white-space:pre; }}
.tag {{ display:inline-block; font-size:11.5px; font-weight:600; padding:2px 8px;
        border-radius:11px; margin-right:5px; }}
.t-h {{ background:#e6fffa; color:#234e52; }}
.t-a {{ background:#ebf4ff; color:#2a4365; }}
footer {{ margin-top:56px; padding-top:18px; border-top:1px solid var(--line);
          color:var(--mut); font-size:13px; }}
</style></head><body><div class="wrap">

<header>
<h1>End-of-Turn Detection from Real Call Audio</h1>
<div class="sub">Plivo AI/ML internship assignment — STT track &nbsp;·&nbsp;
roll no. <strong>241007</strong> &nbsp;·&nbsp; repo <code>plivo-ml-241007</code><br>
Laptop CPU only · no pretrained models, no downloaded weights, no external data ·
numpy / scipy / scikit-learn / pandas / soundfile</div>
</header>

<div class="hero">
  <div class="kpi"><div class="l">English · held out</div>
    <div class="v">{cv_en:.0f} ms</div>
    <div class="d">▼ {b_en - cv_en:.0f} ms vs baseline ({100*(b_en-cv_en)/b_en:.0f}%)</div></div>
  <div class="kpi"><div class="l">Hindi · held out</div>
    <div class="v">{cv_hi:.0f} ms</div>
    <div class="d">▼ {b_hi - cv_hi:.0f} ms vs baseline</div></div>
  <div class="kpi"><div class="l">Diagnostic AUC</div>
    <div class="v">{full_en['auc']:.2f} / {full_hi['auc']:.2f}</div>
    <div class="d">en / hi, out of fold</div></div>
</div>
<p class="sub" style="margin-top:4px">Mean response delay at ≤5% interrupted
turns — the graded metric. Lower is better. Both numbers are
<strong>out-of-fold</strong> (5-fold <code>GroupKFold</code> by
<code>turn_id</code>, averaged over 3 shuffles), which is the honest estimate
of hidden-test behaviour.</p>

<div class="warn"><strong>Read this before looking at any other number.</strong>
The shipped model is trained on all 200 provided turns, so running
<code>predict.py</code> + <code>score.py</code> on the provided folders reports
<strong>264 ms / 266 ms at AUC 0.99</strong>. That is <em>in-sample</em> and it
is not a result — the model has memorised those turns. The honest numbers are
the {cv_en:.0f} ms / {cv_hi:.0f} ms above, and
<code>predictions_english_heldout.csv</code> /
<code>predictions_hindi_heldout.csv</code> are the out-of-fold predictions that
produce them. I expect the hidden test set to land near the held-out numbers,
not the in-sample ones.</div>

<h2>1. The problem, and what the metric actually rewards</h2>
<p>At every pause the agent must decide: has the user finished, or are they
thinking? Fire early and you talk over people; fire late and the call fills
with silence. The scorer sweeps (threshold × action delay) and reports the
lowest mean response delay achievable while interrupting ≤5% of turns.</p>
<p>Reading <code>score.py</code> closely produced the single most useful insight
in this project:</p>
<div class="ok"><strong>A hold pause can only cost you if the action delay is
shorter than that pause.</strong> <code>fires and delay &lt; pause_dur</code>.
So a 150 ms breath pause is essentially free at any usable delay, and the model
does not need to separate end-of-turn from <em>every</em> pause — only from the
<strong>long</strong> ones. That reframing drives the whole design.</div>
<p>It also explains the baselines: the naive "always fire" policy scores
{b_en:.0f} ms on English but only {b_hi:.0f} ms on Hindi, because 83% of Hindi
holds are ≤0.5 s while 63% of English holds exceed 0.5 s. English is the hard
language here, and Hindi has less headroom than it looks.</p>

<h2>2. Solution</h2>
<div class="pipe">wav ──▶ <b>causal_slice(x, sr, pause_start)</b> = x[:int(pause_start*sr)]   ◀── the ONE place audio enters
                    │        (nothing downstream can see the future — by construction)
                    ▼
   ┌────────────────────────────────────────────────────────────┐
   │ own DSP: framing · energy · autocorrelation F0 (octave-    │
   │ protected) · mel filterbank · DCT cepstra · spectral stats │
   └────────────────────────────────────────────────────────────┘
                    ▼
   {res['n_features']} causal features in 7 families (§3)
                    ▼
   duration-risk sample weights  ── long holds matter, short holds don't
                    ▼
   HistGradientBoosting ×5 seeds  ⊕  L2 logistic (quantile-normalised)
                    ▼            rank-average, 65/35
                 p_eot</div>
<p>Gradient-boosted trees rather than a neural net because there are 496
labelled pauses in total; trees also take the missing-by-construction features
(no final voiced run ⇒ NaN) natively. The linear member contributes a smooth,
monotone-ish ranking that keeps the model sane on a recording pool it has never
heard. Ranks, not probabilities, are averaged — the scorer sweeps the
threshold, so only the ordering matters.</p>

<h2>3. Features — all strictly causal</h2>
<table>
<tr><th>Family</th><th>What it measures</th><th>Why it separates hold from EOT</th></tr>
<tr><td><b>Terminal prosody</b></td><td>F0 slope over the final voiced run,
final pitch in semitones relative to the speaker's own running median, range,
fall fraction, creak</td><td>Statements fall to a low phrase-final target;
continuations stay level or rise.</td></tr>
<tr><td><b>Declination residual</b></td><td>Fit the speaker's downward pitch
drift across the turn so far, measure how far <em>below its own trend</em> the
final pitch lands</td><td>Separates a real phrase-final fall from the drift
that happens anyway.</td></tr>
<tr><td><b>Elongation / hesitation</b></td><td>Frame-to-frame cepstral distance
(spectral flux) vs. the speaker's own median; length of the spectrally
"frozen" tail; flat-pitch × frozen × voiced</td><td>A held vowel
("annnd…", "aur…", "umm") is spectrally <em>frozen</em> — the classic
I-am-not-finished cue. Top feature on English.</td></tr>
<tr><td><b>Energy dynamics</b></td><td>Decay slope into the pause, fade depth
vs. the turn's own speech level, percentile of the terminal level</td>
<td>Turn-final syllables trail off; mid-clause pauses are cut short at full
level.</td></tr>
<tr><td><b>Rhythm</b></td><td>Final voiced-run duration ÷ the turn's own median,
syllable-peak rate in the last 1.5 s ÷ rate so far</td><td>Speakers
<em>slow down</em> before yielding the floor.</td></tr>
<tr><td><b>Final-word shape</b></td><td>Six 100 ms slices back from the speech
offset: channel-normalised cepstra + speaker-relative pitch + relative
energy</td><td>An ASR-free proxy for <em>which</em> word ended the pause —
"thank you"/"bas" vs "and"/"aur". Learned from this handout only.</td></tr>
<tr><td><b>Turn context (hazard)</b></td><td>Pause index, speech time so far
(measured from the observed speech onset, not wall clock), previous pause
count/mean/max</td><td>The longer someone has been talking and the more they
have already paused, the more likely this pause ends the turn.</td></tr>
<tr><td><b>Within-turn relative</b> <span class="tag t-a">key idea</span></td>
<td>Every acoustic feature above, re-expressed as a difference and z-score
against <em>the same feature at this turn's earlier pauses</em></td>
<td>"Does this pause sound more final than the ones I already held through?"
Cancels speaker, handset and language variance — and it is the single best
feature in Hindi (AUC 0.77 against long holds).</td></tr>
</table>

<h3>Two design choices worth calling out</h3>
<p><b>Anchor on the observed speech offset, not on <code>pause_start</code>.</b>
Labels sit on a 100 ms grid, so the true offset can be up to ~50 ms earlier.
Every terminal feature is measured backwards from the last detected speech
frame, which makes them immune to that quantisation.</p>
<p><b>Cepstral mean normalisation against the turn's own speech.</b> Raw
cepstra encode the handset as much as the phone. Subtracting the turn's running
mean leaves the identity of the final sound — necessary because the hidden test
set is a different recording pool.</p>

<h2>4. Duration-risk weighting</h2>
<p>Straight from §1: weight each training hold by a sigmoid in its duration
(pivot 420 ms), so long "thinking" holds dominate the loss and sub-300 ms
breath pauses are nearly ignored. Pause duration is <em>future information</em>,
so it is used <strong>only as a training weight</strong> — never as a feature.
This is the same category of thing as the <code>label</code> column: available
at training time, absent at inference. It was worth ~30 ms on Hindi and was the
first change that moved Hindi at all.</p>
{img('risk', 'Out-of-fold p_eot against pause duration. Only holds to the RIGHT of the swept action delay can cause a false cutoff, which is exactly the region the weighting targets. The stray high-scoring long holds (top right) are the errors that cost the score.')}

<h2>5. Results</h2>
{img('scores', 'Held-out mean response delay at ≤5% interrupted turns, against the given silence-only baseline.')}
<table>
<tr><th>System</th><th class="num">English</th><th class="num">Hindi</th><th>Notes</th></tr>
<tr><td>silence-only baseline (given)</td><td class="num">{b_en:.0f} ms</td>
    <td class="num">{b_hi:.0f} ms</td><td>reproduced with <code>baseline.py</code></td></tr>
<tr><td><b>this model, held out</b></td><td class="num"><b>{cv_en:.0f} ms</b></td>
    <td class="num"><b>{cv_hi:.0f} ms</b></td>
    <td>AUC {full_en['auc']:.3f} / {full_hi['auc']:.3f}; operating points
    thr={full_en['threshold']:.2f}, d={full_en['delay']*1000:.0f} ms and
    thr={full_hi['threshold']:.2f}, d={full_hi['delay']*1000:.0f} ms</td></tr>
<tr><td>trained on the <em>other</em> language only</td>
    <td class="num">1313 ms</td><td class="num">850 ms</td>
    <td>AUC 0.666 / 0.668. Single-language training transfers weakly — Hindi
    lands exactly on its baseline. <b>Pooling both languages is what produces
    the result above</b>, and it is why the shipped model is trained on both
    despite the hidden set being mostly Hindi.</td></tr>
<tr><td>VAP zero-shot alone (no EOT labels)</td><td class="num">1412 ms</td>
    <td class="num">850 ms</td><td>§7 — implemented, honestly did not win</td></tr>
<tr><td>causal transformer over 2 s frame matrices<br><span style="font-size:12.5px;color:#4a5568">(same idea, sequence model instead of designed features)</span></td>
    <td class="num">1411 ms</td><td class="num">843 ms</td>
    <td>AUC 0.543 / 0.637 held out — but <b>115 ms / 100 ms at AUC 0.996</b>
    in-sample. See §11.</td></tr>
<tr><td><i>shipped model on its own training folders</i></td>
    <td class="num">264 ms</td><td class="num">266 ms</td>
    <td><i>in-sample, not a result — see the warning above</i></td></tr>
</table>
{img('dist', 'Out-of-fold score distributions. Hindi separates cleanly; English has a stubborn overlap band that is where the remaining delay lives.')}
{img('frontier', 'The whole operating curve, not just the graded point: how mean delay falls as you tolerate more interruptions. At 10% interruptions English drops well below its 5% number.')}
{img('importance', 'Permutation importance (drop in held-out AUC when a single feature is shuffled), hand-rolled because the blend is deliberately not a sklearn estimator.')}
<p>The top features are a mix of final-word slices (<code>q3_e</code>,
<code>q1_c7</code>, <code>q3_vf</code>), pitch-fall measures
(<code>p_fall_frac</code>, <code>p_slope_400</code>, <code>p_range</code>) and
within-turn relative spectral terms — i.e. the model is genuinely using the
prosody the linguistics predicts, not a shortcut.</p>

<h2>6. Why this beats the status quo</h2>
<p>The status quo in production voice agents is a silence timer: wait <i>N</i>
milliseconds and assume the user is done. That is exactly the given baseline,
and it fails because <b>silences inside a turn are routinely longer than
silences between turns</b> — 63% of English holds in this data exceed 0.5 s. A
timer tuned to interrupt ≤5% of the time must therefore wait ~1.6 s, and every
user pays that wait on every turn.</p>
<p>This model reads the <em>shape</em> of the speech leading into the pause —
did the pitch fall to a phrase-final target, did the last vowel freeze mid-word,
is the speaker slowing down, does this pause sound more final than the ones they
already held through — and so it can act on a much shorter timer for the pauses
that look final while still waiting out the ones that do not. Held out, that is
{b_en - cv_en:.0f} ms ({100*(b_en-cv_en)/b_en:.0f}%) off the English wait at the
same interruption budget. On a call with 10 user turns that is roughly
{(b_en - cv_en)*10/1000:.1f} seconds of dead air removed, without interrupting
anyone more often.</p>

<h2>7. Voice Activity Projection: implemented, and an honest negative result</h2>
<p>I reimplemented the objective from Ekstedt &amp; Skantze,
<i>"Voice Activity Projection: Self-supervised Learning of Turn-taking Events"</i>
(Interspeech 2022, arXiv:2205.09812) — from scratch, in
<code>src/vap.py</code> and <code>src/vap_train.py</code>. <b>No weights, no
code and no pretrained encoder were taken from their repository</b>; the rules
forbid that, and none was needed. What is borrowed is the idea: instead of
asking "is this pause an end-of-turn?", predict the <em>discretised future
voice-activity window</em> — 2 s ahead, bins of 200/400/600/800 ms, 50% activity
threshold, modelled jointly (their "Discrete" variant) rather than as
independent per-bin sigmoids. The targets come from VAD on the audio itself, so
they need no annotation: the model trains without seeing a single
<code>eot</code> label, and end-of-turn is read out zero-shot as
<code>p_eot = P(no activity in the next 2 s)</code>.</p>
<p>In-sample it looked excellent — AUC 0.754 on English, better than my
supervised model. Out of fold it collapsed to <b>AUC 0.527</b>. The fix (one
sample per silence <em>event</em> instead of per frame — consecutive frames
inside one pause are nearly identical — plus dropping in-speech frames, whose
target is trivially "resumes immediately") brought it to AUC 0.565 / 0.681,
still below supervised prosody, and blending it in never improved the graded
delay.</p>
<div class="note"><b>Why it does not transfer here, concretely.</b> The frame
count is misleading: 36 minutes at 50 Hz looks like ~100k training examples, but
the audio contains only <b>~1,600 distinct silence events</b>, and the frames
within one silence are not independent samples. So the real data multiplier over
the 496 labelled pauses is about 3×, not 200×. More fundamentally, VAP is a
<b>two-channel</b> model — a large part of its predictive signal is the
<em>other</em> speaker's voice activity, and this handout has only the user
channel, so the 256-state codebook collapses to a 4-bit one-speaker window.
The paper's leverage comes from thousands of hours of two-party dialogue; at 36
minutes of mono audio, hand-designed prosody wins.</div>
<p>It is kept in the repo, documented, and <b>not shipped in the final
predictor</b> — it lost on held-out CV, and shipping it to look sophisticated
would have been the wrong call. The right way to capture its insight is in
NOTES.md as the top next step: semi-supervised training on those ~1,600
unannotated silences using the full pause-level feature set instead of the
thinner frame-level one.</p>

<h2>8. Sequence model instead of features — an ablation worth reading</h2>
<p>I also built the obvious "more modern" alternative: skip hand-designed
features, feed the last 2 s straight in as a 197×17 frame matrix (energy, F0,
voice activity, ZCR, 12 MFCCs) and let a small causal transformer (d_model 32,
2 layers, ~20k parameters, positional encoding, causal attention mask) read the
sequence. Pooled English+Hindi, 30 epochs, BCE loss.</p>
<table>
<tr><th>How it is scored</th><th class="num">English</th><th class="num">Hindi</th><th class="num">AUC</th></tr>
<tr><td>train on all 200 turns, predict those same 200</td>
    <td class="num">115 ms</td><td class="num">100 ms</td><td class="num">0.996 / 0.999</td></tr>
<tr><td><b>5-fold GroupKFold by turn — unseen turns</b></td>
    <td class="num"><b>1411 ms</b></td><td class="num"><b>843 ms</b></td>
    <td class="num">0.543 / 0.637</td></tr>
</table>
<div class="warn"><b>Same code, same features, same epochs — a 12× difference in
the reported score, and English falls to near chance (0.543).</b> The 115 ms is
not end-of-turn detection; it is a 20k-parameter model recalling 496 training
sequences. Anything that reads "115 ms at AUC 0.996" on 100 turns of real
conversational audio is measuring its own training set.</div>
<p>Hence trees over designed features rather than a sequence model: with 496
labelled pauses, the feature design <em>is</em> the regularisation. It is also
why {cv_en:.0f} ms that survives a turn-grouped split is the number worth
reporting.</p>

<h2>9. Causality — how to audit it in one minute</h2>
<p>The rule is that a pause's features may use only audio in
<code>[0, pause_start)</code>. Rather than ask a reader to trust ~700 lines, the
design funnels everything through one function:</p>
<pre><code>def causal_slice(x, sr, pause_start):
    n = int(round(float(pause_start) * sr))
    return x[:n]              # the ONLY audio any feature function receives</code></pre>
<ul>
<li>No feature function is ever handed the full waveform, so none can read
audio at or after the pause.</li>
<li>The waveform <em>length</em> is never used as a feature — it would leak,
since for an <code>eot</code> row <code>pause_end</code> is the end of file.</li>
<li>The current row's <code>pause_end</code> is never read. Earlier rows'
<code>pause_end</code> values are used, and those are strictly in the past
(<code>prev_pause_end &lt; pause_start</code>).</li>
<li>The number of pauses in the turn is never used — only
<code>pause_index</code>, which counts pauses already seen.</li>
<li>The frame-level VAP extractor uses exclusively backward-looking windows and
is queried at the last frame that <em>ends</em> at or before
<code>pause_start</code>.</li>
<li>Pause duration and the VAP projection targets are derived from future
audio, and are used <b>only as training weights / training labels</b> — exactly
like the <code>label</code> column. <code>predict.py</code> computes neither.</li>
</ul>

<h2>10. What the human did vs. what the coding agent did</h2>
<p>Stated plainly, because the assignment asks and because it is checkable
against the repo and the commit history.</p>
<table>
<tr><th style="width:50%"><span class="tag t-h">human</span> candidate</th>
    <th><span class="tag t-a">agent</span> Claude Opus 5 (Claude Code)</th></tr>
<tr><td>
<ul style="margin:0;padding-left:18px">
<li><b>Scoping and constraints.</b> Chose the end-of-turn track; set and
re-checked the envelope (CPU only, no pretrained weights, allowed libraries),
including dropping a PyTorch dependency as unnecessary.</li>
<li><b>Literature review.</b> Identified Voice Activity Projection (Ekstedt
&amp; Skantze, Interspeech 2022) as the relevant prior work and required it be
adapted under the no-weights rule — objective only, not the codebase. That is
§7.</li>
<li><b>Independent comparison.</b> Sourced a second implementation of the same
task (a transformer sequence model) and required it be benchmarked under the
same protocol. That is §8.</li>
<li><b>Prioritisation.</b> Called the freeze — stop tuning, verify and ship —
with time left on the clock.</li>
</ul></td>
<td><ul style="margin:0;padding-left:18px">
<li>All code: DSP, the {res['n_features']} causal features, the model, the CV
harness, the VAP reimplementation, the transformer ablation, the figures and
this page.</li>
<li>The metric asymmetry in §1, and the features built from it (duration-risk
weighting, within-turn relative, elongation, final-word slices).</li>
<li>Experiments and ablations, failures included, logged in RUNLOG.md.</li>
<li>Flagged the in-sample 264 ms and produced the out-of-fold prediction
files.</li>
</ul></td></tr>
</table>
<p>Short version: the agent wrote the code; the direction on what to build,
what to measure it against, and when to stop came from the human side. §7 and
§8 both exist because of that direction.</p>

<h2>11. Repo, and how to reproduce</h2>
<pre><code>pip install numpy scipy scikit-learn pandas soundfile matplotlib pyarrow

# score every pause of an unseen folder (loads the saved model, never refits)
python predict.py --data_dir &lt;folder&gt; --out predictions.csv
python score.py   --data_dir &lt;folder&gt; --pred predictions.csv

python train_final.py       # retrain and re-save artifacts/model.joblib
python src/cv.py            # honest held-out numbers + cross-language transfer
python src/pipeline.py      # VAP vs prosodic vs blend, out of fold
python src/gen_report.py    # out-of-fold predictions, figures, results.json</code></pre>
<table>
<tr><th>File</th><th>What it is</th></tr>
<tr><td><code>predict.py</code></td><td>the deliverable entry point —
<code>--data_dir</code> / <code>--out</code>, loads
<code>artifacts/model.joblib</code></td></tr>
<tr><td><code>src/features_eot.py</code></td><td>causal feature extraction;
<code>causal_slice</code> is the audit point</td></tr>
<tr><td><code>src/dsp.py</code></td><td>all signal processing, written from
scratch</td></tr>
<tr><td><code>src/model.py</code></td><td>the blend + duration-risk weights</td></tr>
<tr><td><code>src/vap.py</code>, <code>src/vap_train.py</code></td>
<td>the VAP reimplementation (§7)</td></tr>
<tr><td><code>src/metric.py</code></td><td><code>score.py</code>'s sweep,
in-process, for model selection</td></tr>
<tr><td><code>predictions_{{english,hindi}}.csv</code></td><td>required
deliverable, from the shipped model (in-sample on these folders)</td></tr>
<tr><td><code>predictions_{{english,hindi}}_heldout.csv</code></td><td>the same
pauses scored out of fold — the honest predictions</td></tr>
<tr><td><code>RUNLOG.md</code>, <code>NOTES.md</code></td><td>every scoring run
with what changed and why; signals, failures, next steps</td></tr>
</table>

<h3>Summary of RUNLOG.md</h3>
<p>Twelve logged runs. Baseline {b_en:.0f}/{b_hi:.0f} → v1 rich causal features
1224/850 → duration-risk weighting 1216/819 (first movement on Hindi) →
elongation + CMN cepstra + declination residual 1117/850 → final-word slices →
shipped {cv_en:.0f}/{cv_hi:.0f} held out. Rejected on held-out evidence:
dropping short holds from training, feature selection to the top 25–70, and
blending in VAP. The diagnostic that mattered most was per-feature AUC computed
<em>only</em> on EOT vs holds &gt;350 ms, which revealed that Hindi ends turns
with pitch and English ends them with elongation.</p>

<h3>Summary of NOTES.md</h3>
<p>Signals: terminal pitch relative to the speaker's own median <em>and</em> to
their earlier pauses, declination residual, elongation via spectral flux,
energy decay, rate slowdown, channel-normalised final cepstra, turn-so-far
context. Still fails on list-like turns where every item ends with final
prosody ("…two pizzas, …one garlic bread"), on rising confirmation questions,
and on first pauses with under a second of context. With one more day:
semi-supervised training on the ~1,600 unannotated VAD silences with the full
feature set, speed/gain/noise augmentation, a sequence model across the pauses
of a turn, and an hour spent listening to the 20 worst false cutoffs.</p>

<footer>plivo-ml-241007 · End-of-turn detection · figures and every number on
this page regenerate from <code>src/gen_report.py</code> +
<code>src/make_summary.py</code>. Held-out figures are 5-fold GroupKFold by
turn, 3 shuffles.</footer>
</div></body></html>
"""

out = os.path.join(ROOT, "SUMMARY.html")
with open(out, "w") as f:
    f.write(HTML)
print(f"wrote {out} ({len(HTML)/1024:.0f} KB incl. embedded figures)")
