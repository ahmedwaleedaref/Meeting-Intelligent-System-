# CI and Experiment Tracking: Findings and Decisions

## 1. CI for frozen data

I don't think we should implement the full data-dependent CI checks yet. The core ICSI data and frozen releases are not expected to change during the current baseline stage, so parsing the full corpus and checking the releases on every pull request would slow things down without regularly validating changed data.

I suggest we revisit CI once the baseline pipeline and its tests are clearer. We can then run lightweight tests frequently and reserve full-corpus and reproducibility checks for when they are useful, such as on demand, nightly, or after relevant data-processing changes.

I still think a versioned Docker data image is a feasible option later. It could provide the raw NXT data, metadata, parsed C1 output, and frozen releases without repeatedly downloading Git LFS files. I just don't think we need to build it at this stage.

## 2. Experiment tracking

I recommend that we use Weights & Biases (W&B) through the academic workspace. This gives us shared experiment tracking and comparison without requiring us to maintain an MLflow server and its backing storage. I want Git to remain our durable, reviewable record, with W&B complementing it rather than replacing it.

We should manage one workspace for the project and add the team as collaborators. From what I saw, the workspace provides around 20 GB of storage and broad collaborator access.

## 3. How I propose we integrate W&B

I don't think we need to settle every directory name now. The important thing is to agree on the responsibilities and keep them consistent with our existing repository protocol.

- **Run entry point:** When we introduce `scripts/run.py`, it should resolve the experiment configuration, create the W&B run, and log its metrics and metadata.
- **Experiment definitions:** Keep configurations in Git, using our existing Hydra setup where appropriate, so changes remain reviewable and traceable.
- **Local outputs:** Store run-specific files under `runs/<run_id>/` and keep generated outputs out of Git. Share or synchronize them according to the repository protocol.
- **Results and conclusions:** Commit the results table under `results/` and keep a short `NOTE.md` for each experiment. These records should remain useful even if we change tracking tools later.
- **W&B project:** Use one shared project for baseline experiments. Reuse the same `run_id` for the local run directory, results-table row, and W&B run so we can trace each result across all three.

For every run, I suggest recording the resolved configuration, Git commit, data version or fingerprint, random seed, fold or split, task, experiment name, and whether the test set was used. We should log training and validation history as metrics, keep final evaluation metrics in the run summary, and upload `predictions.jsonl` as a W&B Artifact when we need it for paired comparisons such as the planned bootstrap analysis.

We should also support offline runs with `WANDB_MODE=offline` and sync them later, so a network issue doesn't interrupt an experiment. API keys should stay in environment variables or GitHub secrets, never in the repository.
