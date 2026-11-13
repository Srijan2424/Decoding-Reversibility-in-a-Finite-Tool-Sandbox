# Remaining research work: status on 13 November 2026

## Completed in this update
- Prepared 1,200 blinded same-example RQ2 questions per model, tied to the original dataset hash and saved validation-selected probe predictions.
- Verified strict label parsing, invalid-answer accounting and agreement calculations with controlled tests. No model answers have been generated for this new experiment.
- Audited the pinned ScienceWorld source. The object-tree export omits genetic/pollination properties, stopwatch counters and agent waiting state. This proves it is not a complete simulator serialization in general; it does not prove those fields affected the earlier particular cases.
- Prepared an offline demonstration replaying genuine saved Qwen agent calls, with independent transition/score checks.

## Next GPU session: same-example RQ2
Upload `43_main_dataset_rq2.py` to a Colab GPU notebook using the existing Drive project and dependencies. For Qwen, run:

```python
%run /content/43_main_dataset_rq2.py --model qwen --mode generate
%run /content/43_main_dataset_rq2.py --model qwen --mode analyze
```

In a separate runtime, repeat with `--model llama`. The Llama token must be available in Colab Secrets as `HF_TOKEN`. All 1,200 rows are queried once per model; held-out comparisons reuse the three original splits (600 unique held-out IDs), without changing layer or regularisation choices. Interruptions resume from saved answers. `--limit 20` limits newly generated answers for an interface check, but does not count as a completed experiment. Do not tune prompts based on held-out answers. The protocol is explicitly a late extension after main test results were visible.

The script accepts `--project` for local paths. Default is `/content/drive/MyDrive/ADNN_Irreversibility`. No need to rerun hidden-state extraction or main probes. Analysis includes accuracy, balanced accuracy, stated/probe agreement and Cohen's kappa, and 1,000 scenario-cluster bootstrap intervals. Only four test groups per split limit precision. Binary evaluation retains partial/invalid answers as errors, rather than dropping them.

## ScienceWorld: required order of work
1. Define exactly which physical state is to be restored and which bookkeeping variables are excluded, with justification. Adding all historical counters to an exact-restoration test would make ordinary actions trivially irreversible.
2. Extend/validate the export for the chosen physical state, including subclass fields, new objects, drainage destinations and inventory. Audit each relevant object type. Store the modified source, new JAR hash and tests separately from the original simulator.
3. Validate deterministic reset-and-replay and action identity for every candidate. Exclude ambiguous/self-pour cases. Never freeze time or omit thermal differences merely to obtain convenient labels.
4. Use a resumable exhaustive breadth-first search over the declared legal action set for horizons 1, 2 and 3. A witness establishes possible recovery; lack of a witness establishes impossibility only after complete enumeration. Search timeouts remain unresolved. A restricted action set requires a clearly restricted claim.
5. Freeze the interpretation and action set before fresh evaluation. Establish the positive reversible, permitted-partial and irreversible cases. Do not fill a dataset with no-ops to reach a row target.
6. Only after the gate above passes, collect at least 800 additional distinct accepted examples so the combined total reaches 2,000. Keep task/variation groups and minimal pairs together in splits. Count each state-action pair once, not once per horizon.
7. Build a new blinded annotation sample covering both environments. Preserve the earlier 100-row answers as a separate completed audit. Human and AI annotate independently before seeing reference labels.
8. Extract both models' ScienceWorld activations when a GPU is available. Train sandbox-to-ScienceWorld and ScienceWorld-to-sandbox probes with shared label definitions and training-only preprocessing. Use training/validation decisions without target-test tuning.
9. Run the stated-answer comparison on the added examples and update every report table from audited outputs.

## What cannot be claimed yet
No new ScienceWorld dataset labels, transfer scores or RQ2 model results were produced by preparing these scripts. A correct pipeline is not a completed experiment. With no GPU today, the RQ2 inference and new-environment representation measurements remain pending. The current report retains these limitations.
