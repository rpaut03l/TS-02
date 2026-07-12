# DLOPS_07 — W&B Sweeps, Course Version (wandb.init/log/sweep/agent, define_metric)

[Hub](DLOPS_EXAM_00_Hub.md) | [Prev: TensorBoard](DLOPS_06_TensorBoard.md) | [Next: W&B Official Sweeps](DLOPS_08_WandB_Sweeps_Official.md)

> Story: in module 6 YOU were the experiment runner — 8 runs, 8 writers, your
> fingers on the keyboard. Weights & Biases hires a robot intern for you: you
> write down the recipe space ("try lr between 0.0001 and 0.1, batch 16/32/64"),
> and an **agent** cooks combination after combination, uploading every tasting
> note to a cloud dashboard. Your job shrinks to two things: describe the
> search space, and write ONE train function the robot can call.

> **Cross-repo reference (TS-01):** hyperparameter search here formalizes the
> estimation intuition from TS-01 →
> [Parameter-Estimations-Guide](https://github.com/rpaut03l/TS-01/tree/main/ML/Parameter-Estimations-Guide) |
> [Regression](https://github.com/rpaut03l/TS-01/tree/main/ML/Regression)

## table of contents
- [1. setup — install, login, keys](#1-setup--install-login-keys)
- [2. the wandb run lifecycle](#2-the-wandb-run-lifecycle)
- [3. wandb.config — the single source of truth](#3-wandbconfig--the-single-source-of-truth)
- [4. step 1 — the sweep config](#4-step-1--the-sweep-config)
- [5. step 2 — the training function](#5-step-2--the-training-function)
- [6. step 3 — create the sweep and run the agent](#6-step-3--create-the-sweep-and-run-the-agent)
- [7. define_metric — best vs last](#7-define_metric--best-vs-last)
- [8. reading results — api and dashboard](#8-reading-results--api-and-dashboard)
- [9. hyperband early termination](#9-hyperband-early-termination)
- [10. getting a usable final model](#10-getting-a-usable-final-model)
- [11. mnemonics](#11-mnemonics)
- [12. cheatsheet](#12-cheatsheet)
- [13. exam hacks and trap watch](#13-exam-hacks-and-trap-watch)

---

## 1. setup — install, login, keys

```bash
pip install wandb
```

```python
import wandb
wandb.login()        # first time: paste API key from wandb.ai/authorize
```

- The key is cached (`~/.netrc`), so login is once per machine.
- In Colab, `wandb.login()` prompts inline; in scripts you can also export
  `WANDB_API_KEY=...` as an env var (the automation-friendly way — mention
  it in "how would you run this in CI?" style questions).
- Free tier note from real experience: projects under a hidden team org need
  a `?nw=nwuser<username>` query param to share views — dashboards are
  account-scoped, unlike TensorBoard's local files.

Mental model vs TensorBoard:

```
TensorBoard:  code -> event files on YOUR disk -> local dashboard
W&B:          code -> HTTPS -> wandb.ai cloud  -> web dashboard
                                   |
                       + sweep SERVER that hands out configs to agents
```

The sweep server is the new ingredient — it's what makes automated search possible.

[back to top](#table-of-contents)

## 2. the wandb run lifecycle

Everything in W&B happens inside a **run** — one training attempt:

```python
run = wandb.init(
    project="dlops-fmnist",              # dashboard bucket for related runs
    name="baseline-adam-lr1e3",          # optional human-readable run name
    config={"lr": 1e-3, "batch_size": 32, "epochs": 5},   # hyperparams
)

for epoch in range(cfg.epochs):
    ...
    wandb.log({"train_loss": tr_loss,    # each log() = one row of metrics
               "val_loss": va_loss,
               "epoch": epoch})

wandb.finish()                           # closes the run, flushes uploads
```

Line by line:
- `wandb.init` starts the run, opens the network channel, snapshots your
  config, and even records git state + system metrics (GPU util, RAM) for
  free — reproducibility metadata you'd otherwise lose.
- `config` = the settings; `log` = the outcomes. Settings vs outcomes is the
  same split as TensorBoard's add_hparams — but logged live, per step.
- `wandb.finish()` is writer.close()'s cousin. The context-manager form
  handles it automatically and is what the class notebook uses inside sweeps:

```python
with wandb.init(config=config) as run:
    ...                                  # finish() called for you on exit
```

**Mnemonic — ILAF: Init, Log, (Artifact later,) Finish** — the run lifecycle.

[back to top](#table-of-contents)

## 3. wandb.config — the single source of truth

The most important discipline in this module. Inside a run, NEVER hard-code a
hyperparameter — read everything from `wandb.config`:

```python
with wandb.init(config=hyperparameters):
    cfg = wandb.config                       # attribute-style dict
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)
    model = build_model(cfg.hidden_units)
    optimizer = build_optimizer(model, cfg.optimizer, cfg.lr)
    for epoch in range(cfg.epochs):
        ...
```

Why this matters (the exam "why" behind the pattern):
1. In a normal run, config is what YOU passed — nothing changes.
2. In a SWEEP, the agent OVERRIDES config with the combo it wants tested.
   Because your code reads cfg.*, the same function serves both worlds
   with zero edits. That's the entire trick that makes sweeps plug-and-play.

Helper the class wrote — optimizer chosen by config string:

```python
def build_optimizer(model, optimizer_name, lr):
    if optimizer_name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "rmsprop":
        return torch.optim.RMSprop(model.parameters(), lr=lr)
```

Now "optimizer" is just another sweepable string parameter — architecture
choices become hyperparameters too (same pattern extends to activation type,
number of layers, dropout p...).

[back to top](#table-of-contents)

## 4. step 1 — the sweep config

The recipe space, as a plain dict. The three mandatory keys:

```python
sweep_config = {
    "method": "random",                      # HOW to search
    "metric": {                              # WHAT to optimize
        "name": "val_loss",                  # must match a wandb.log key!
        "goal": "minimize",                  # or "maximize"
    },
    "parameters": {                          # WHERE to search
        "lr": {
            "distribution": "log_uniform_values",
            "min": 1e-4,
            "max": 1e-1,
        },
        "batch_size": {"values": [16, 32, 64]},
        "epochs":     {"value": 5},                       # FIXED value
        "optimizer":  {"values": ["adam", "sgd", "rmsprop"]},
        "hidden_units": {"values": [64, 128, 256]},
    },
}
```

Parameter spec grammar — the three shapes:

```
{"value": x}            fixed — same in every run (still logged)
{"values": [a, b, c]}   categorical — pick one from the list
{"min": a, "max": b}    continuous range — sample within
  + optional "distribution": "uniform" | "log_uniform_values" | ...
```

Why `log_uniform_values` for lr (a genuinely favorite theory question):
learning rates matter on a LOG scale — the difference between 0.0001 and
0.001 (10x) is as meaningful as 0.01 to 0.1. Uniform sampling of [1e-4, 1e-1]
would spend ~90% of samples above 0.01; log-uniform spreads them evenly
across the decades.

The three methods (deep dive on bayes in module 8):

```
grid    exhaustively try every combination of DISCRETE values
random  independent random draws — strong baseline, handles continuous
bayes   builds a model of metric-vs-hyperparams; proposes promising points
```

**Mnemonic — MMP: Method, Metric, Parameters** — the config skeleton.
**"GRB: Grid tries all, Random rolls dice, Bayes takes notes."**

[back to top](#table-of-contents)

## 5. step 2 — the training function

One self-contained function the agent can call repeatedly. The class version,
assembled from the same building blocks as modules 5-6:

```python
def train_sweep():
    with wandb.init() as run:               # agent INJECTS the config here
        cfg = wandb.config

        # data — batch size from config
        train_loader = DataLoader(train_data, batch_size=cfg.batch_size,
                                  shuffle=True)
        val_loader = DataLoader(val_data, batch_size=cfg.batch_size,
                                shuffle=False)

        # model — width from config
        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, cfg.hidden_units), nn.ReLU(),
            nn.Linear(cfg.hidden_units, 10),
        ).to(device)

        loss_fn = nn.CrossEntropyLoss()
        optimizer = build_optimizer(model, cfg.optimizer, cfg.lr)

        for epoch in range(cfg.epochs):
            model.train()
            for X, y in train_loader:
                X, y = X.to(device), y.to(device)
                loss = loss_fn(model(X), y)
                optimizer.zero_grad(); loss.backward(); optimizer.step()

            # validation — the metric the sweep optimizes
            model.eval(); val_loss, correct = 0, 0
            with torch.inference_mode():
                for X, y in val_loader:
                    X, y = X.to(device), y.to(device)
                    logits = model(X)
                    val_loss += loss_fn(logits, y).item()
                    correct += (logits.argmax(1) == y).sum().item()
            val_loss /= len(val_loader)
            val_acc = correct / len(val_loader.dataset)

            wandb.log({"val_loss": val_loss,          # NAME MUST MATCH
                       "val_acc": val_acc,            # the metric config!
                       "epoch": epoch})
```

Three graded details:
1. `wandb.init()` with NO arguments inside a sweep — the agent supplies
   project and config. (You can pass defaults; agent values win.)
2. The logged key `"val_loss"` must EXACTLY match `metric.name` in the
   config, or the sweep optimizes nothing (silent failure — trap #1).
3. Log per EPOCH so early-termination schedulers (section 9) have
   checkpoints to judge runs by.

[back to top](#table-of-contents)

## 6. step 3 — create the sweep and run the agent

```python
# 3a. register the sweep on the W&B server -> returns an id
sweep_id = wandb.sweep(sweep_config, project="dlops-sweeps")
# printed: sweep URL + "Run sweep agent with: wandb agent <entity/project/id>"

# 3b. start an agent: pulls a config, calls train_sweep, repeats `count` times
wandb.agent(sweep_id, function=train_sweep, count=10)
```

What actually happens, in a picture:

```
                 +--------------------------+
                 |  W&B SWEEP SERVER        |
                 |  knows: config space,    |
                 |  method, results so far  |
                 +-----+-----------+--------+
        "give me next" |           | "give me next"
                       v           v
                 [ agent 1 ]   [ agent 2 ]     <- can be different machines!
                 train_sweep() train_sweep()
                       |           |
                 wandb.log results back up ----> dashboard
```

- `count=10` = this agent runs 10 combos then stops. Omit count for grid
  sweeps (agent stops when the grid is exhausted) — with random/bayes and no
  count, it runs FOREVER (trap!).
- Parallelism for free: run `wandb.agent(sweep_id, ...)` on your Mac AND on
  Colab simultaneously — the server hands each a different combo. Distributed
  hyperparameter search with zero extra code (a genuinely great exam point).
- CLI equivalent: `wandb sweep config.yaml` then `wandb agent <id>` — same
  three steps, shell edition.

[back to top](#table-of-contents)

## 7. define_metric — best vs last

Problem: a run's "summary" value for val_loss is by default the LAST logged
value. But the best epoch might be in the middle (overfitting after!).

```python
wandb.define_metric("val_loss", summary="min")   # summary = best-so-far min
wandb.define_metric("val_acc",  summary="max")
```

- Call it right after wandb.init(), before logging.
- Now the runs table's `val_loss.min` column ranks runs by their BEST epoch —
  which is what you actually deploy (with early stopping/checkpointing).
- Also usable to declare custom x-axes: `wandb.define_metric("epoch")` then
  `wandb.define_metric("val_*", step_metric="epoch")` — plots against epoch
  instead of wandb's internal step counter.

[back to top](#table-of-contents)

## 8. reading results — api and dashboard

Dashboard (know the two plot names — module 8 details them):
- **Parallel coordinates plot** — one line per run across hyperparameter axes,
  ending at the metric; winning bundles pop out visually.
- **Hyperparameter importance plot** — which knob correlates most with the
  metric (lr almost always dominates).

Programmatic access — the class used `wandb.Api()` to fish out the best run:

```python
api = wandb.Api()
sweep = api.sweep(f"{entity}/{project}/{sweep_id}")
best_run = sweep.best_run()                    # by the config's metric
print(best_run.name, best_run.config, best_run.summary["val_loss"])
```

- `sweep.best_run()` respects your metric+goal — no manual sorting.
- `run.config` gives you the winning hyperparameters as a dict — feed them
  straight into a final training (next section).

[back to top](#table-of-contents)

## 9. hyperband early termination

Budget saver: why let a clearly-bad run finish all its epochs?

```python
sweep_config["early_terminate"] = {
    "type": "hyperband",
    "min_iter": 3,        # first judgment checkpoint after 3 logged epochs
    # "s": 2,             # optional: number of bracket halvings
}
```

How Hyperband thinks (interview-grade explanation):
- It sets checkpoints ("brackets") at min_iter, min_iter*eta, min_iter*eta^2...
  (eta defaults to 3: epochs 3, 9, 27...).
- At each checkpoint it compares all runs that reached it and KILLS the
  bottom fraction — successive halving.
- Effect: bad configs die at epoch 3; only promising ones earn full budgets.
  Total compute for a sweep drops severalfold with almost no loss in
  best-found quality.
- Requirement: the metric must be logged regularly (per epoch) — another
  reason for the epoch-wise wandb.log discipline.

```
epochs ->  1  2  3 | 4 ... 9 | 10 ... 27
run A      . . x   killed at bracket 1 (worst third)
run B      . . .   . . . x   killed at bracket 2
run C      . . .   . . . .   . . . survives -> full training
```

[back to top](#table-of-contents)

## 10. getting a usable final model

The sweep found settings, but sweep runs usually don't save weights. The
class's final move: retrain once with the winner's config, save properly.

```python
best_config = best_run.config                     # from wandb.Api()

with wandb.init(project="dlops-final", config=best_config):
    cfg = wandb.config
    model = build_model(cfg).to(device)
    optimizer = build_optimizer(model, cfg.optimizer, cfg.lr)
    train_full(model, optimizer, cfg)             # maybe MORE epochs now
    torch.save(model.state_dict(), "best_model.pth")
    wandb.save("best_model.pth")                  # attach file to the run
```

- `wandb.save(path)` uploads the file into the run's Files tab — the
  primitive that module 9's Artifacts formalize with versioning + lineage.
- Ops flow to narrate in a long answer: sweep (search) -> best_run.config
  (decide) -> final run (train properly) -> saved + logged weights (deploy).

[back to top](#table-of-contents)

## 11. mnemonics

- **ILAF** — run lifecycle: Init, Log, Artifact, Finish.
- **MMP** — sweep config skeleton: Method, Metric, Parameters.
- **"GRB: Grid tries all, Random rolls dice, Bayes takes notes."**
- **"cfg or it didn't happen"** — every hyperparameter reads from wandb.config.
- **"Name match or no match"** — logged key == metric.name, exactly.
- **"Hyperband = talent show: bottom third eliminated each round."**
- **"log scale for lr — decades, not decimals."**

[back to top](#table-of-contents)

## 12. cheatsheet

```
SETUP        pip install wandb ; wandb.login()  (key from wandb.ai/authorize)
RUN          with wandb.init(project, config=...) as run:  ... wandb.log({...})
CONFIG       cfg = wandb.config ; cfg.lr, cfg.batch_size  (agent overrides)
SWEEP CFG    {"method": grid|random|bayes,
              "metric": {"name": "val_loss", "goal": "minimize"},
              "parameters": {p: {"value"|"values"|"min"+"max"[+distribution]}}}
LR SPACE     distribution: log_uniform_values, min 1e-4, max 1e-1
CREATE       sweep_id = wandb.sweep(sweep_config, project=...)
AGENT        wandb.agent(sweep_id, function=train_sweep, count=10)
             (no count + random/bayes = runs forever; multiple agents = parallel)
BEST METRIC  wandb.define_metric("val_loss", summary="min")
EARLY STOP   sweep_config["early_terminate"] = {"type":"hyperband","min_iter":3}
RESULTS      api = wandb.Api(); sweep.best_run(); best_run.config
FINAL        retrain with best config -> torch.save + wandb.save
```

[back to top](#table-of-contents)

## 13. exam hacks and trap watch

1. Logged metric name != metric.name in config -> sweep "works" but optimizes
   nothing. The #1 silent failure — check spelling first.
2. `{"value": x}` (fixed) vs `{"values": [...]}` (choices) — one letter, big MCQ.
3. Random/bayes agent without `count` never stops. Grid stops itself.
4. `wandb.init()` inside a sweep takes no config argument fight — agent wins.
5. Hard-coded batch_size inside train_sweep = that parameter silently never
   sweeps. Everything through cfg.
6. define_metric must be called BEFORE the logs it affects.
7. Hyperband needs regular metric logging (per epoch) to have checkpoints.
8. grid method with a continuous {"min","max"} param -> error/undefined — grid
   needs discrete values.
9. Multiple agents on one sweep_id = parallel search, no code change — free
   marks when asked "how to speed up a sweep?"
10. wandb.log step counter is global per run — log epoch explicitly as a field
    (and/or define_metric step_metric) to plot per-epoch.

[back to top](#table-of-contents)

---
[Hub](DLOPS_EXAM_00_Hub.md) | [Prev: TensorBoard](DLOPS_06_TensorBoard.md) | [Next: W&B Official Sweeps](DLOPS_08_WandB_Sweeps_Official.md)
