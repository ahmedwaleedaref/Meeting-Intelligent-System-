# Repository Protocol

This document defines how work enters the repository. Its goal is that anyone can open
one task and know everything that was done in it, why, and with what result, without
asking the person who did it.

The words **must**, **must not** and **should** are used deliberately.

---

## 1. Core units

| Unit | What it is | Where it lives |
|---|---|---|
| **Task** | A unit of project work with a goal (e.g. `C1` encoder baseline). | `tasks/<ID>-<slug>/` |
| **Experiment** | One question asked inside a task (e.g. "is the LR too high?"). | `tasks/<ID>-<slug>/experiments/expNN_<slug>/` |
| **Run** | One execution of an experiment (one seed, one fold, one grid point). | `runs/<run_id>/` (not in git) + one row in the results table |
| **Concept note** | A short explanation of a method or concept used by a task. | `concepts/` |

An experiment always belongs to exactly one task. A run always belongs to exactly one experiment.

---

## 2. Tasks

### 2.1 Naming
`tasks/<ID>-<slug>/`, e.g. `tasks/C1-encoder-baseline/`. The ID comes from the task list.

### 2.2 Ownership
- Every task has **one owner**, listed in the root `README.md` task table.
- Others may contribute to a task, but the owner keeps its README correct and allocates experiment numbers.

### 2.3 The task README is the single entry point
`tasks/<ID>-<slug>/README.md` must contain:
- **Goal**: what the task must produce.
- **Status**: not started / in progress / done.
- **Decisions**: every decision taken in the task, one line each, with a one-line reason.
- **Experiments**: an index table with one row per experiment: number, question, result, conclusion.

The README describes the **current state** of the task. It is not a diary. Detail belongs in the experiment notes, and the README links to them.

### 2.4 Task extension
When work does not fit an existing task:
- If it serves the goal of an existing task, it is an **extension** of that task. Add it as experiments in that task and update the task README.
- If it has a new goal, it is a **new task**. Copy `tasks/_template/`, give it an ID, and add a row to the root `README.md` task table.
- New tasks must be approved by the project lead before work starts.

---

## 3. Experiments

### 3.1 When to create one
Create an experiment whenever you run something whose result you might report, compare, or base a decision on. Quick throwaway checks belong in `notebooks/` (see §8).

### 3.2 Naming and numbering
- `expNN_<slug>`, e.g. `exp03_lr-sweep`. `NN` is sequential within the task; `slug` says what it is.
- The task owner allocates numbers. Before starting work, **reserve the number** by adding the experiment's row to the task README index on `main`.

### 3.3 Contents
Each experiment folder contains exactly:
- `config.yaml`: the path of the base config it extends, plus **only** the fields this experiment changes.
- `NOTE.md`: question, change vs baseline, run IDs, result, conclusion.

### 3.4 Experiments do not contain code
- All experiments run the shared code in `src/`. They differ **only in config**.
- If an experiment needs new behaviour (a new loss, a new input format), that behaviour is added to `src/` behind a config option (see §5), and the experiment turns it on in its config.

### 3.5 Definition of done
An experiment is **done** only when:
1. `NOTE.md` has a result and a conclusion, and
2. its row in the task README index is filled.

Negative and null results must be written up the same way. "LR 1e-5 did not help" is a result.

---

## 4. Runs

### 4.1 Launching
- Runs are launched **only** through `scripts/run.py`, pointing at an experiment's `config.yaml`.
- `run.py` refuses to start if the working tree has uncommitted changes. Every run must correspond to a commit.

### 4.2 What every run records (automatically)
`runs/<run_id>/`:

| File | Content |
|---|---|
| `config.yaml` | the fully resolved config |
| `meta.json` | git commit, data version, seed, fold, start/end time, hardware |
| `history.jsonl` | per-step/epoch training loss, validation metrics, learning rate |
| `metrics.json` | final metrics, and the selected epoch if any |
| `predictions.jsonl` | predictions in the harness schema |

`run.py` also appends **one row per run** to the results table (`results/`), with the run ID, task, experiment and final metrics.

### 4.3 Rules
- Run directories are never edited or deleted by hand. A failed or wrong run stays recorded; mark it in the experiment `NOTE.md`.
- Run outputs (`runs/`) are not committed to git. They are synced to shared storage, keyed by run ID.
- The results table is committed to git.

---

## 5. Code (`src/`)

- `src/` holds the only copy of shared code: data loading, input building, models, training, evaluation.
- Changes to `src/` go to `main` **only through a pull request reviewed by one other member**.
- Changes must not silently change the behaviour of existing configs. New behaviour is added behind a config option whose default keeps the old behaviour, so every past experiment can still be re-run.
- Code must not be copied into task or experiment folders.

---

## 6. Data

- The dataset is versioned (`v1`, `v2`, …) and stored outside git. `data/` is local only.
- A dataset version is **never edited in place**. Any change produces a new version, documented in the data task README (A10).
- Every run records the data version it used (`meta.json`).

---

## 7. Evaluation and the test set

- Model selection (hyperparameters, epochs, prompts, architecture choices) uses **cross-validation folds only**.
- The test set is read **only by the evaluation harness**, through an explicit flag that is recorded in the run metadata.
- Test evaluation happens only for final results, with the project lead's approval.
- Best-epoch selection uses validation data, never test, and the selected epoch is recorded in `metrics.json`.

---

## 8. Notebooks

- `notebooks/` is for exploration, sanity checks and figures.
- A number may appear in a task README, a NOTE, a report or the paper **only if a run produced it**. Notebook output alone does not count.

---

## 9. Documentation

| Document | Contains | Does not contain |
|---|---|---|
| Root `README.md` | project map, how to run, task table | task details |
| Task `README.md` | goal, status, decisions, experiment index | concept explanations, run-by-run history |
| Experiment `NOTE.md` | question, change, run IDs, result, conclusion | general background |
| `concepts/*.md` | a concept the project uses, explained briefly | anything no task uses |

Rules:
- Task docs and notes **must not explain concepts**. They link to a concept note instead.
- A concept note exists **only if at least one task README or NOTE links to it**.
- Concept notes are short and focused on what this project needs; link external sources for the rest.

---

## 10. Branches and commits

- `main` always reflects reviewed code and finished documentation.
- Work happens on branches named `<taskID>/<short-description>`, e.g. `C1/exp03-lr-sweep`.
- Experiment folders (config + NOTE) and task README updates may be merged without review. Changes to `src/` require review (§5).

---

## 11. Checklists

### Starting an experiment
- [ ] Number reserved in the task README index on `main`
- [ ] Branch created: `<taskID>/expNN-<slug>`
- [ ] `experiments/expNN_<slug>/config.yaml` written (base + changes only)
- [ ] `NOTE.md` has the question and the change vs baseline
- [ ] Any needed code is in `src/` via a reviewed PR
- [ ] Working tree clean before launching `run.py`

### Finishing an experiment
- [ ] Run IDs listed in `NOTE.md`
- [ ] Result and conclusion written (including negative results)
- [ ] Task README index row filled
- [ ] Any decision that follows added to the task README "Decisions"
- [ ] Merged to `main`