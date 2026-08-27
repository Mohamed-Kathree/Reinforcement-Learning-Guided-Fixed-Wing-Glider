# Why `reward_audit_phase5_policy_STALE_CHECKPOINT.json` is not evidence

This file (originally `reward_audit_phase5_policy.json`) was produced during the Phase 5
throughput smoke-check by running `scratch/reward_audit.py --policy` against
`checkpoints/ppo_glider_final.zip` + `checkpoints/vecnormalize_final.pkl`.

At the time it was generated, `scratch/reward_audit.py --policy` did not yet record which
checkpoint it had audited (see `RLGlider_Phase6_Barrier_Unit_Fix_Spec.md` §C.1). Reconstructing
provenance after the fact: no Phase 5 training run had happened by that point in the session
(`Rl-Glider-Context-File-Code-V11-Phase5-Reward-Rescaling.md` states this explicitly), so the only
checkpoints on disk were V10's stale artefacts — roughly 10% into an aborted 2M-step smoke-check,
trained against the **pre-Phase-5 reward function** (the one with the unbounded reachability
barrier, Defect A, and the numerically dead Gaussian precision term, Defect B). The audited
policy's own numbers are consistent with this: it lands a mean 163 m from home at Stage 0, far
worse than the scripted baseline's 26 m, and shows a barrier-dominated reward budget matching what
an untrained-on-the-current-reward policy would produce.

**Consequence:** Phase 5's acceptance criteria 5.5 and 5.6 (reward-budget terminal/barrier share
under a *trained* policy) were never actually verified — they were measured against a policy that
never saw the Phase 5 reward function at all. This should be read as **unverified, not failed**.

This file is kept for historical reference only. Do not use it to draw any conclusion about how a
policy trained on the current (Phase 5+Phase 6) reward function behaves — re-run
`scratch/reward_audit.py --policy <checkpoint> --vecnorm <pkl>` against a freshly trained
checkpoint instead. As of Phase 6, `--policy` runs record the audited checkpoint's absolute path,
mtime, and a truncated SHA-256 directly in the output JSON's `provenance` field, so this
reconstruction-after-the-fact should not be necessary again.
