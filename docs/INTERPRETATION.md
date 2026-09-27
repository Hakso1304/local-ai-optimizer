# Benchmark Result Interpretation Guide (v1)

Author: Fable (orchestrator). Status: normative for `src/core/interpret/` — every rule
below has an id and is meant to be evaluated by the app, not only read by humans.
Numbers are calibration-derived from this machine (RX 9070 XT 16 GB, 31 GB RAM,
llama.cpp b11208 Vulkan) and the model families measured on 2026-09-27; they are
starting points, all live in `rules.v1.json`, and must be adjustable.

The app must never print an interpretation without the measured evidence it rests on.
Every insight has: `severity` (info | note | warn | critical), `metric`, `text`,
`evidence[]` (metric values with provenance), and, where sensible, `action`.

---

## 0. Reading order (what to look at first)

1. **Did it run at all?** status per rung: pass / degraded / fail (+ failureKind).
2. **Where is the practical context ceiling** and *why* (limitedBy: memory | spill |
   cliff | declared | failure). This is the single most important line.
3. **Quality with its confidence band.** A quality number without `n` and a band is not
   a result.
4. **Speed at the context you will actually use** (recommendedCtx), not at 2K.
5. **Memory margin** at that context (VRAM headroom, shared-GPU spill, RAM floor).
6. Only then compare candidates, and only within the same workload.

Rule I-0.1: an insight panel must lead with items 2 and 3; speed comes after.

---

## 1. Provenance vocabulary (how much to trust a number)

| Kind | Meaning | Trust |
|---|---|---|
| MEASURED | read from llama-server timings, PDH counters, wall clock | full |
| ESTIMATED | derived by formula (memory estimates, wall-clock TPS fallback, quality prior) | directional only; never decides a ranking against a MEASURED value |
| DECLARED | reported by the model file / runtime (context length, parameter count) | true of the file, not of your machine |
| UNAVAILABLE | not observable here (AMD temperature/power, <1 s steps with no telemetry row) | neutral; never rendered as 0 |

Rule I-1.1 (`prov.estimated-in-ranking`, warn): if any component that decided the
winner is ESTIMATED, the recommendation is *provisional* and says which one.
Rule I-1.2 (`prov.unavailable-memory`, note): if peak VRAM/RAM is UNAVAILABLE for
a rung, say "memory for this rung was not observed (step shorter than the 1 s sampler)".

---

## 2. Context ladder and the practical ceiling

Definitions: the **declared** context is what the file says (e.g. 131072). The
**practical ceiling** is the largest rung that passed cleanly. **Degraded** means it
ran but a cliff rule fired. **Fail** means OOM / device lost / crash / timeout /
guard abort / config drift.

Rule I-2.1 (`ctx.ceiling`, info, always shown): "Practical context: 16K (declared
131K). Limited by <limitedBy> at <next rung>: <first cliff/failure reason with numbers>."

Rule I-2.2 (`ctx.spill`, warn): any rung with adjusted shared-GPU spill > 256 MiB →
"Above <ctx> the model no longer fits in VRAM: <spill> GiB moved to system memory;
decode fell <a> → <b> t/s." Action: "use ≤ <last clean rung>, or KV cache q8_0, or a
smaller quantization."

Rule I-2.3 (`ctx.memory-bound`, note): ceiling limited by *memory* with no cliff →
"Larger contexts were not attempted: estimated VRAM <x> > budget <y>." Action: "KV
q8_0 halves KV; heavy mode can put KV in RAM (slower)."

Rule I-2.4 (`ctx.declared-vs-practical`, warn): practical < 25 % of declared → "This
model advertises <declared> but on this machine <practical> is the usable limit."

Rule I-2.5 (`ctx.required-not-met`, critical): requiredContext set and no config
reaches it → lead insight; list each model's ceiling and limitedBy.

Rule I-2.6 (`ctx.transient-dip`, note): a decode drop that recovers on the next rung
is reported as "transient dip at <ctx> (likely contention), not a cliff".

---

## 3. Speed: which number matters for which use

**Decode (generation) t/s** — how fast text streams out. **Prefill t/s** — how fast the
prompt is read. **TTFT** — wait before the first token; at a given context it is
≈ prompt tokens ÷ prefill t/s. **Effective answer t/s** (thinking models) — answer
tokens ÷ total seconds including reasoning; this is what the user *feels*.

### 3.1 Decode bands (per config, at recommendedCtx)

| Band | decode t/s | Reads as |
|---|---|---|
| unusable | < 3 | slower than reading; batch-only |
| patient | 3 – 8 | acceptable for one-off high-quality answers (Maximum Quality) |
| usable | 8 – 20 | fine for large-scale coding/agent work where quality dominates |
| comfortable | 20 – 40 | everyday chat/coding |
| snappy | 40 – 100 | interactive assistants |
| instant | > 100 | small models; speed no longer a differentiator |

Rule I-3.1 (`speed.decode-band`, info): state the band and the workload's gate:
"12.6 t/s — usable (Coding gate 10 t/s; your floor <user value> if set)."
Rule I-3.2 (`speed.thinking-effective`, warn): thinking on and effective answer t/s
< 50 % of raw decode → "reasoning consumes <r> tokens per answer; effective <e> t/s."

### 3.2 TTFT bands (at the context you will use)

| Band | TTFT | Reads as |
|---|---|---|
| immediate | < 1 s | chat feel |
| short wait | 1 – 5 s | fine for coding turns |
| noticeable | 5 – 20 s | acceptable for long-context work if expected |
| long | 20 – 60 s | batch / "go get coffee"; only with a required context |
| impractical | > 60 s | reject unless the user explicitly required this context |

Rule I-3.3 (`speed.ttft-band`, info/warn): band + "at <ctx> with a <prompt tokens>-token
prompt". Warn when TTFT band is worse than the workload tolerance; when the user
required the context, phrase as *accepted because required*.

### 3.3 Prefill

Rule I-3.4 (`speed.prefill-scaling`, note): prefill t/s falling faster than
0.5× per context doubling → "prompt processing slows super-linearly; expect TTFT to
grow faster than context".

### 3.4 Partial offload

Rule I-3.5 (`speed.partial-offload`, warn, always on expectDegraded configs):
"<n>/<m> layers on GPU — <decode> t/s vs <full-offload alternative> t/s; CPU <cpu %>."
Add "KV on CPU (-nkvo) is slower than dropping ~5 layers at short context" when that
rung exists (measured 2026-09-27: dense 7.4 vs 10.7, MoE 27.0 vs 38.5).
Rule I-3.6 (`speed.moe-note`, info): expertCount > 0 → "Mixture-of-experts: only
~<active> B parameters run per token, so partial offload costs less than for a dense
model (measured 3.6–4× faster than a dense 27B at similar GPU share)."

---

## 4. Memory: margins, not just fit

Rule I-4.1 (`mem.headroom`, info/warn): VRAM headroom at recommendedCtx =
budget − peak dedicated. < 0.5 GiB → warn "little headroom; other GPU apps will push
this into shared memory". 
Rule I-4.2 (`mem.in-use-at-plan`, note): VRAM in use by other apps at planning time
> 1.5 GiB → "planned while <x> GiB was already in use; results may improve on an idle GPU."
Rule I-4.3 (`mem.ram-floor`, warn): min available RAM during a rung < floor + 1 GiB →
"came within <d> GiB of the RAM safety floor". guard_abort → critical with the reason.
Rule I-4.4 (`mem.mmap-note`, info): full-offload configs show RAM drop ≈ file size:
"this is the file cache (mmap), reclaimable — not a leak".
Rule I-4.5 (`mem.wddm-83`, note): per-PID dedicated stalls at ≈ 83 % of VRAM on WDDM
before spilling; "VRAM 95 % is not reachable on Windows, spill starts earlier".

---

## 5. Quality: what the number is and is not

Q = 100 · Σ_c W_c · passRate_c / Σ W_c over categories with results (binary pass per
item; suite qb-1.1.0, 17 items; v2 planned ~60 items + seeded variants).

Rule I-5.1 (`quality.band`, info): report `Q ± ci95 (n items[, s samples])` and the
per-category pass rates. Bands: < 40 weak · 40–60 limited · 60–80 solid · 80–95
strong · > 95 saturated (suite too easy to separate models at this level).
Rule I-5.2 (`quality.ci-overlap`, warn): two top candidates whose bands overlap →
"quality difference <d> is inside the confidence band (n=<n>) — not decisive; speed and
memory decided." Action: "run Thorough quality mode".
Rule I-5.3 (`quality.category-gap`, note): any category ≤ 33 % → name it ("reasoning
1/4") and, for coding workloads, warn if `coding` ≤ 66 %.
Rule I-5.4 (`quality.thinking-off`, note): thinking-capable model scored with thinking
off → say so; if a thinking gen-config was measured, show both with effective speed.
Rule I-5.5 (`quality.estimated`, warn): quality is a prior (params × quant), not
measured → "no measured quality; ranking on quality is provisional."
Rule I-5.6 (`quality.small-suite`, note): n < 30 → "small suite; ±<ci> is wide".
Rule I-5.7 (`quality.not-a-leaderboard`, info, once per panel): "Quality is a
relative signal for *these* candidates on *this* suite, not a leaderboard score."

---

## 6. Stability and reproducibility

Rule I-6.1 (`stab.rep-variance`, warn): reps differ > 15 % in decode → "measurements
noisy (contention or thermal); rerun on an idle system". 
Rule I-6.2 (`stab.versions`, warn): runs mix runtime/suite versions → say which.
Rule I-6.3 (`stab.failures`, warn/critical): any oom / device_lost / crash /
config_drift in the session → list per config with the stderr classification;
device_lost → critical ("GPU reset; results after it are suspect").
Rule I-6.4 (`stab.cold-run`, note): `warm=false` rows are excluded from speed scoring.

---

## 7. Comparing candidates (the "why" block)

Rule I-7.1: comparisons are only valid within one workload and one machine snapshot.
Rule I-7.2: every why-not sentence carries numbers for quality delta, speed at the
reference context, ceiling, and the gate that failed (if any) — in that order.
Rule I-7.3 (`cmp.quality-vs-speed`, info): when the winner is slower but higher
quality: "chosen for quality over speed: <Q_w ± ci> vs <Q_r ± ci>; <decode_w> vs
<decode_r> t/s (<k>× slower)". When the winner is faster but lower quality: state that
the quality gap is inside the band (I-5.2) or that the workload weights speed.
Rule I-7.4 (`cmp.same-model-quant`, note): same model, different quantization: quality
delta ≤ ci → "quantization difference not measurable on this suite; prefer the smaller".

---

## 8. Generation configuration (thinking / effort / temperature)

Rule I-8.1 (`gen.best-config`, info): per model, name the best gen config and the
delta it produced: "thinking on (effort low, T=1.0): Q 88 vs 71 off; answers 2.3× slower".
Rule I-8.2 (`gen.stochastic`, note): T > 0 → "sampled (seeded), n=<s> per item; expect
±<ci> run-to-run".
Rule I-8.3 (`gen.effort-saturation`, note): higher effort adds reasoning tokens without
quality gain (within ci) → recommend the lower effort.

---

## 9. What to do next (actions vocabulary)

Actions are short, imperative and tied to a rule:
- `use-context <ctx>` — stay at or below the last clean rung
- `enable-kv-q8` — halve KV memory (small quality risk, re-measure)
- `enable-heavy-mode` — allow partial offload for models that don't fit
- `lower-required-context` / `raise-min-decode`
- `try-smaller-quant <suggestion>` — e.g. Q4_K_M → IQ4_XS / Q3_K_XL from the same repo
- `try-thinking-config` — run the gen-config search
- `run-thorough-quality` — v2 suite + repeated samples
- `rerun-idle` — measurements were contended
- `download <model>` — when the fastest/best-quality alternative is a known file

Rule I-9.1: an insight with severity warn/critical must carry at least one action.

---

## 10. Panel layout (for the UI)

Dashboard card: headline · practical ceiling line (I-2.1) · quality with band (I-5.1)
· decode band (I-3.1) · at most 2 warn/critical insights.
Results → "Interpretation" section: all insights grouped by section 2–8, each with its
evidence expandable (metric, value, provenance, rung). Actions rendered as buttons where
the app can execute them (enable heavy mode → Benchmark preset; download → Hub page).

---

## 11. Calibration provenance for the thresholds

Decode/TTFT bands: qualitative, chosen against measured 8B (110→52 t/s to 64K; TTFT
0.5 s@2K … 32 s@64K), 14B (spill at 32K: 51→26 t/s), Qwen3.8-27B partial (12.6→10.8
t/s, TTFT 1.9–6.7 s), Gemma-4 MoE partial (~46–50 t/s). Spill threshold 256 MiB and
WDDM 83 % stall: measured 2026-09-27. Quality bands: judgment; revisit after suite v2.
