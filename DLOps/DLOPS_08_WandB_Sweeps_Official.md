# DLOPS_08 — W&B Sweeps, Official Notebook (sweep_config: method/metric/parameters, MNIST pipeline)

[Hub](DLOPS_EXAM_00_Hub.md) | [Prev: Sweeps Course](DLOPS_07_WandB_Sweeps_Course.md) | [Next: Artifacts](DLOPS_09_WandB_Artifacts.md)

> Story: module 7 was the course's home-grown sweep; this is W&B's OWN official
> tutorial (13c) that the TA ran in class — same three-step skeleton, but with
> the canonical structure the docs (and exams copying the docs) use: a nested
> pipeline of build_dataset / build_network / build_optimizer / train_epoch,
> a LogSoftmax + NLL loss combo, and per-BATCH logging. Learn the deltas from
> module 7 and you can answer questions phrased in either dialect.

> **Cross-repo reference (TS-01):** grid/random/bayes search space thinking
> pairs with TS-01 →
> [Parameter-Estimations-Guide](https://github.com/rpaut03l/TS-01/tree/main/ML/Parameter-Estimations-Guide)

## table of contents
- [1. the three-step skeleton, official phrasing](#1-the-three-step-skeleton-official-phrasing)
- [2. picking a method — grid, random, bayes in depth](#2-picking-a-method--grid-random-bayes-in-depth)
- [3. naming the parameters — full grammar](#3-naming-the-parameters--full-grammar)
- [4. the pipeline functions, line by line](#4-the-pipeline-functions-line-by-line)
- [5. logsoftmax + nll — the alternative loss pairing](#5-logsoftmax--nll--the-alternative-loss-pairing)
- [6. per-batch logging and the running example](#6-per-batch-logging-and-the-running-example)
- [7. initialize and run — sweep_id and agent](#7-initialize-and-run--sweep_id-and-agent)
- [8. visualizations — parallel coordinates and importance](#8-visualizations--parallel-coordinates-and-importance)
- [9. module 7 vs module 8 — the diff table](#9-module-7-vs-module-8--the-diff-table)
- [10. mnemonics](#10-mnemonics)
- [11. cheatsheet](#11-cheatsheet)
- [12. exam hacks and trap watch](#12-exam-hacks-and-trap-watch)

---

## 1. the three-step skeleton, official phrasing

The official notebook names the three steps exactly like this — use these
labels when the question sounds doc-flavored:

```
Step 1. DEFINE the sweep      -> write sweep_config (the what)
Step 2. INITIALIZE the sweep  -> sweep_id = wandb.sweep(sweep_config)  (the where)
Step 3. RUN the sweep agent   -> wandb.agent(sweep_id, function=train) (the go)
```

And its two-sentence definition worth quoting: a sweep ties together a
STRATEGY for trying hyperparameter combos (grid/random/bayes) with the CODE
that evaluates them (your training function). The server coordinates;
agents execute.

```
   sweep_config ---> wandb.sweep() ---> sweep server
                                             |
                              wandb.agent(sweep_id, train)
                                   |    |    |
                                 run1 run2 run3 ... (each = wandb.init + logs)
```

[back to top](#table-of-contents)

## 2. picking a method — grid, random, bayes in depth

The official notebook spends real time here — expect a compare/contrast Q.

**grid** — Cartesian product of every `values` list.
- Pros: exhaustive, reproducible, embarrassingly parallel.
- Cons: explodes combinatorially (4 params x 5 values = 625 runs); cannot
  handle continuous ranges at all.
- Use when: few discrete knobs, cheap runs.

**random** — every run draws each parameter independently from its
distribution.
- Pros: handles continuous + categorical; anytime-stoppable; famously
  beats grid at equal budget (Bergstra & Bengio's argument: with k important
  dims out of many, random covers each important dim's range with EVERY
  sample, grid wastes samples on duplicated values of the important dim).
- Cons: no learning from results; needs a count or it never stops.
- Use when: default choice, medium budgets, unknown landscape.

**bayes** — Bayesian optimization: fit a surrogate model P(metric | config)
(typically a Gaussian process) over completed runs; pick the next config
maximizing expected improvement; repeat.
- Pros: sample-efficient — best results per run for expensive trainings.
- Cons: sequential at heart (parallelism dilutes it); overhead grows with run
  count; works best with fewer, continuous dimensions.
- Use when: each run is costly (big models), <20 dims.

```
budget of 9 runs over 2 hyperparams (x = tried):

grid (3x3 lattice)    random                 bayes (converging)
x . . x . . x         . x . . . x .          . . . . x . .
. . . . . . .         x . . x . . .          . . . x x x .
x . . x . . x         . . x . . . x          . . . . x . .
covers lattice only   covers ranges          clusters near optimum
```

**Mnemonic — "GRB: Grid tries all, Random rolls dice, Bayes takes notes."**

[back to top](#table-of-contents)

## 3. naming the parameters — full grammar

The official notebook's config (MNIST MLP) — memorize this exact shape:

```python
sweep_config = {
    "method": "random",
    "metric": {"name": "loss", "goal": "minimize"},
    "parameters": {
        "optimizer":     {"values": ["adam", "sgd"]},
        "fc_layer_size": {"values": [128, 256, 512]},
        "dropout":       {"values": [0.3, 0.4, 0.5]},
        "epochs":        {"value": 1},                    # fixed, still logged
        "learning_rate": {"distribution": "uniform",
                          "min": 0, "max": 0.1},
        "batch_size":    {"distribution": "q_log_uniform_values",
                          "q": 8, "min": 32, "max": 256},
    },
}
import pprint; pprint.pprint(sweep_config)                # the notebook's habit
```

Two distribution specs beyond module 7 — decode them:
- `"uniform", min, max` — flat continuous sampling.
- `"q_log_uniform_values", q=8, min=32, max=256` — sample log-uniformly,
  then QUANTIZE to multiples of q: batch sizes come out as 32, 40, 48...256.
  Log-ish spread + hardware-friendly rounding in one spec.

Grammar summary (superset of module 7's three shapes):

```
{"value": x}                                   constant
{"values": [...]}                              categorical (optionally + probabilities)
{"distribution": d, "min": a, "max": b}        continuous;  d in
     uniform | log_uniform_values | q_uniform | q_log_uniform_values (needs q) | ...
{"distribution": "normal", "mu": m, "sigma": s}  gaussian around a guess
```

Also legal: nested sweeps via YAML files, and `parameters` inside
`parameters` for grouped configs — name-drop only, don't memorize.

[back to top](#table-of-contents)

## 4. the pipeline functions, line by line

The official notebook decomposes train into four builders — the modular style
exams love to ask you to complete:

```python
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms

def build_dataset(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),   # MNIST's real mean/std!
    ])
    dataset = datasets.MNIST(".", train=True, download=True,
                             transform=transform)
    sub_dataset = torch.utils.data.Subset(
        dataset, indices=range(0, len(dataset), 5))   # every 5th sample = 20%
    loader = torch.utils.data.DataLoader(sub_dataset, batch_size=batch_size)
    return loader

def build_network(fc_layer_size, dropout):
    network = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, fc_layer_size), nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(fc_layer_size, 10),
        nn.LogSoftmax(dim=1),                          # NOTE: log-probs out!
    )
    return network.to(device)

def build_optimizer(network, optimizer, learning_rate):
    if optimizer == "sgd":
        return optim.SGD(network.parameters(), lr=learning_rate, momentum=0.9)
    elif optimizer == "adam":
        return optim.Adam(network.parameters(), lr=learning_rate)

def train_epoch(network, loader, optimizer):
    cumu_loss = 0
    for _, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        loss = F.nll_loss(network(data), target)       # pairs with LogSoftmax
        cumu_loss += loss.item()
        loss.backward()
        optimizer.step()
        wandb.log({"batch loss": loss.item()})         # per-BATCH logging
    return cumu_loss / len(loader)
```

Notable details, each a potential question:
- `Normalize((0.1307,), (0.3081,))` — MNIST's actual dataset mean/std, the
  "proper" normalization vs the lazy (0.5, 0.5). One channel -> one-tuples.
- `Subset(dataset, range(0, len, 5))` — stride-5 slicing = 20% subset =
  "start small" discipline applied inside a sweep (each of many runs stays cheap).
- Dropout position: after the ReLU of the hidden layer — sweepable p.
- Builders take PLAIN arguments; only the orchestrator touches wandb.config —
  clean separation (testable functions, one config boundary).

The orchestrator gluing it together:

```python
def train(config=None):
    with wandb.init(config=config):
        config = wandb.config
        loader = build_dataset(config.batch_size)
        network = build_network(config.fc_layer_size, config.dropout)
        optimizer = build_optimizer(network, config.optimizer,
                                    config.learning_rate)
        for epoch in range(config.epochs):
            avg_loss = train_epoch(network, loader, optimizer)
            wandb.log({"loss": avg_loss, "epoch": epoch})    # matches metric!
```

[back to top](#table-of-contents)

## 5. logsoftmax + nll — the alternative loss pairing

Module 2 said "logits + CrossEntropyLoss." This notebook uses the other legal
pairing — know BOTH and the equivalence:

```
Pairing A (course default):
  model outputs LOGITS  ->  nn.CrossEntropyLoss(logits, labels)
                              (log_softmax + NLL fused inside)

Pairing B (official notebook):
  model ends with nn.LogSoftmax(dim=1)  ->  F.nll_loss(log_probs, labels)

A == B mathematically.  CE(logits) = NLL(LogSoftmax(logits))
```

Why anyone uses B: the model directly emits log-probabilities — handy when
you also want them at inference (just exp() for probs). Why A is the usual
default: fused kernel = slightly faster/numerically safer, and you can't
accidentally double-apply softmax.

The ILLEGAL combos (trap MCQs):
- Softmax (not Log) + NLLLoss -> NLL expects LOG-probs; wrong scale.
- LogSoftmax + CrossEntropyLoss -> log_softmax applied twice.

**Mnemonic — "CE eats raw; NLL eats logs."**

[back to top](#table-of-contents)

## 6. per-batch logging and the running example

Two logging granularities in one run — both visible in the dashboard:

```python
wandb.log({"batch loss": loss.item()})     # inside train_epoch: every batch
wandb.log({"loss": avg_loss, "epoch": epoch})   # per epoch: the sweep metric
```

- Per-batch curves are noisy but show WITHIN-epoch dynamics (lr too high
  shows up in seconds, not epochs).
- The sweep still optimizes the smooth per-epoch `"loss"` — noisy metrics
  make bayes and hyperband dumber, so optimize the averaged one.
- wandb's x-axis is its internal step counter (increments every log call);
  logging `"epoch"` as a field lets you switch the axis in the UI.

[back to top](#table-of-contents)

## 7. initialize and run — sweep_id and agent

```python
sweep_id = wandb.sweep(sweep_config, project="pytorch-sweeps-demo")
wandb.agent(sweep_id, train, count=5)
```

- The printed sweep page URL is mission control: live table of runs, the two
  key plots (next section), and controls to PAUSE/RESUME/STOP the sweep —
  server-side state means you can kill your notebook and resume agents later.
- `count=5` here because it's a demo; real usage 20-100+ for random.
- Same parallel-agents trick as module 7 — the id is the rendezvous point.

[back to top](#table-of-contents)

## 8. visualizations — parallel coordinates and importance

The two named plots the official notebook showcases — describable in words
because exams ask "what does X show?":

**Parallel coordinates plot**

```
 lr        batch    dropout   fc_size   |  loss
0.1 ─┐                                  |  high
     ├────╲   64 ──╲  0.5 ─╲   512 ─╲   |
0.01 ┤     ╲ 32 ────╳──────╳────────╳───┼─ low   <- trace the low-loss lines
     ├──────╳                           |          backwards to read the
0.001┘       256 ─╱  0.3 ─╱   128 ─╱    |          winning recipe
```

- One polyline per RUN, passing through its value on each hyperparameter
  axis, terminating at the metric axis (color-coded).
- Read it by eye: where do the good-colored lines bunch on each axis?
  That bunching IS the answer to "which settings work."

**Hyperparameter importance plot**
- For each hyperparameter: a correlation and an importance score (from a
  random forest trained ON the sweep's runs — hyperparams as features, metric
  as target — a delightfully meta detail worth writing).
- Tells you which knobs to keep sweeping and which to freeze — the input to
  your NEXT, narrower sweep. Iterative refinement is the intended workflow.

[back to top](#table-of-contents)

## 9. module 7 vs module 8 — the diff table

One table to reconcile both notebooks (answers phrased either way score):

| aspect | module 7 (course) | module 8 (official) |
|---|---|---|
| dataset | FashionMNIST | MNIST, stride-5 Subset (20%) |
| normalize | (0.5,) style | real mean/std (0.1307, 0.3081) |
| loss pairing | logits + CrossEntropyLoss | LogSoftmax + F.nll_loss |
| model knobs | hidden_units, optimizer, lr, batch | fc_layer_size, dropout, optimizer, lr, batch |
| lr distribution | log_uniform_values | uniform (0..0.1) |
| batch distribution | values list | q_log_uniform_values (q=8) |
| logging | per epoch (val_loss/val_acc) | per batch + per epoch avg loss |
| extras | define_metric, Api().best_run, hyperband | pprint config, the two plots |
| structure | one train_sweep function | build_dataset/network/optimizer + train_epoch |

Same skeleton (MMP config -> wandb.sweep -> wandb.agent); different garnish.

[back to top](#table-of-contents)

## 10. mnemonics

- **"Define, Initialize, Run"** — the official three-step naming.
- **GRB** — Grid tries all, Random rolls dice, Bayes takes notes.
- **"q means quantized: log-spread, then snap to multiples of q."**
- **"CE eats raw; NLL eats logs."** — the two legal loss pairings.
- **"Builders take arguments; only train() touches config."** — clean structure.
- **"Parallel coordinates: follow the good lines home."**
- **"Importance plot = a forest judging your knobs."**

[back to top](#table-of-contents)

## 11. cheatsheet

```
STEPS        1 define sweep_config -> 2 wandb.sweep -> 3 wandb.agent(id, train, count)
METHODS      grid (discrete-only, exhaustive) | random (default, anytime)
             | bayes (surrogate model, expensive runs, few dims)
PARAM SPECS  value | values | distribution+min/max
             uniform | log_uniform_values | q_log_uniform_values (+q) | normal(mu,sigma)
MNIST NORM   Normalize((0.1307,), (0.3081,))       real stats, 1-channel tuples
SUBSET       Subset(ds, range(0, len(ds), 5))      20% for cheap sweep runs
NETWORK      Flatten -> Linear(784,fc) -> ReLU -> Dropout(p) -> Linear(fc,10)
             -> LogSoftmax(dim=1)
LOSS         F.nll_loss(log_probs, target)   == CE(logits, target)
LOGGING      per batch: {"batch loss": ...} ; per epoch: {"loss": avg, "epoch": e}
PLOTS        parallel coordinates (runs as lines) | importance (forest-ranked knobs)
WORKFLOW     broad random sweep -> read importance -> narrow sweep -> final train
```

[back to top](#table-of-contents)

## 12. exam hacks and trap watch

1. NLLLoss with plain Softmax (not LogSoftmax) = wrong — NLL wants LOG-probs.
2. LogSoftmax + CrossEntropyLoss = double log_softmax — the mirror trap.
3. `q_log_uniform_values` without `q` -> invalid spec; q defines the rounding grid.
4. MNIST normalize tuples have ONE element — it's grayscale; (0.5,0.5,0.5) is
   the RGB reflex error.
5. `Subset(ds, range(0, len(ds), 5))` — stride slicing, NOT the first 20%;
   preserves class mix across the dataset (why stride beats head-slice).
6. Metric name in this notebook is `"loss"` — the per-epoch average, not
   "batch loss"; sweep optimizes what metric.name says, verbatim.
7. Grid + a distribution-typed param -> error; grid needs values lists.
8. bayes with 30+ hyperparameters degrades toward random — dimensionality point.
9. The importance plot is computed BY a random forest on run data — the meta
   detail that separates read-the-docs answers from guessers.
10. Sweeps can be paused/resumed from the web UI — server-side state; agents
    are disposable workers.

[back to top](#table-of-contents)

---
[Hub](DLOPS_EXAM_00_Hub.md) | [Prev: Sweeps Course](DLOPS_07_WandB_Sweeps_Course.md) | [Next: Artifacts](DLOPS_09_WandB_Artifacts.md)
