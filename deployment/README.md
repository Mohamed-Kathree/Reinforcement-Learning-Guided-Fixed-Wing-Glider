# deployment/

The single current best-trained-model triple, tracked directly in git (an explicit exception to
the `checkpoints/`/`*.zip` gitignore rules — see `.gitignore` for why). This is so any device with
this repo cloned gets the deployable model via a plain `git pull`, without a manual out-of-band
download.

## `best_model/`

From the 20M-step reward-fix retrain (commit `c55119f`), evaluated in
`Rl-Glider-Context-File-Code-V14-RewardFix-Retrain-Final-Outcome.md` (`Context_For_Claude/`,
sibling of this checkout).

- `best_model.zip` — the policy, selected by `EvalCallback` on Stage-3 evaluation performance.
- `vecnormalize_best.pkl` — its matching observation/reward normalisation stats. **Never use the
  `.zip` without this** — an un-normalised policy is not the same policy.
- `curriculum_at_best_approx.json` — the curriculum state nearest to when this checkpoint was
  saved. Approximate, not exact (see the V14 write-up for why); not needed to *use* the model, only
  potentially relevant if resuming training from it.

Load it the same way any other checkpoint pair is loaded, e.g.:
```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

model = PPO.load("deployment/best_model/best_model.zip")
# wrap your env with VecNormalize.load("deployment/best_model/vecnormalize_best.pkl", venv)
```

**Do not overwrite this directory with a future run's `best_model.zip` without updating the
context-file reference to it** — this is meant to be the one deliberately-archived deployment
artifact, not a place training scripts write to automatically.
