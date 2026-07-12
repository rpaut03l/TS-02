# DLOPS_09 — W&B Artifacts (data + model versioning pipeline)

[Hub](DLOPS_EXAM_00_Hub.md) | [Prev: Official Sweeps](DLOPS_08_WandB_Sweeps_Official.md) | [Next: Distributed](DLOPS_10_Distributed_Training.md)

> Story: you version your CODE with git — but your model was born from code
> PLUS data PLUS a previous model's weights. When prod misbehaves at 2 AM,
> "which dataset version trained the model we deployed?" must have an answer.
> **Artifacts** are git-for-heavy-things: every dataset and model becomes a
> named, versioned, checksummed package, and because runs record what they
> USED and what they PRODUCED, you get a full family tree (lineage) of your
> pipeline. This is the most "Ops" notebook in the whole course.

> **Cross-repo reference (TS-01):** for the "what have I already tried"
> ledger habit this module formalizes, see the running gap tracker in TS-01 →
> [ml_master_gap_index.md](https://github.com/rpaut03l/TS-01/blob/main/ML/ml_master_gap_index.md)

## table of contents
- [1. what is an artifact and why care](#1-what-is-an-artifact-and-why-care)
- [2. the demo pipeline map](#2-the-demo-pipeline-map)
- [3. stage 1 — log a dataset, line by line](#3-stage-1--log-a-dataset-line-by-line)
- [4. stage 2 — use and preprocess, line by line](#4-stage-2--use-and-preprocess-line-by-line)
- [5. stage 3 — train and log the model](#5-stage-3--train-and-log-the-model)
- [6. stage 4 — evaluate a logged model](#6-stage-4--evaluate-a-logged-model)
- [7. versioning mechanics — hashes, aliases, metadata](#7-versioning-mechanics--hashes-aliases-metadata)
- [8. the graph view — lineage](#8-the-graph-view--lineage)
- [9. api mini-reference — every call](#9-api-mini-reference--every-call)
- [10. mnemonics](#10-mnemonics)
- [11. cheatsheet](#11-cheatsheet)
- [12. exam hacks and trap watch](#12-exam-hacks-and-trap-watch)

---

## 1. what is an artifact and why care

Definition to write verbatim: **an Artifact is a named, versioned collection
of files (dataset, model, or anything), logged by a run and usable by other
runs, with automatic checksum-based versioning and recorded lineage.**

The problems it kills (each one an "explain why versioning matters" bullet):
1. **Reproducibility** — rerun training on the EXACT bytes of dataset v3.
2. **Provenance** — prod model -> which run made it -> which data it consumed.
3. **Deduplication** — content hashing means re-logging unchanged files
   uploads nothing and creates no new version.
4. **Collaboration** — teammates `use_artifact("dataset:latest")` instead of
   "the zip I emailed you in March."

Type vs name (a small but tested distinction):

```python
wandb.Artifact(name="mnist-raw", type="dataset")
#              ^ identity within project   ^ category: dataset | model | ...
```

Same name = one version lineage (v0, v1, ...). Type groups artifacts in the UI
and semantically labels graph nodes.

[back to top](#table-of-contents)

## 2. the demo pipeline map

The notebook builds a 4-stage MNIST pipeline; every stage is one RUN with a
`job_type`, consuming and/or producing artifacts:

```
[run: load-data]  ──produces──>  (artifact: mnist-raw, dataset)
                                        │
[run: preprocess-data] ──uses──────────┘
                  ──produces──>  (artifact: mnist-preprocess, dataset)
                                        │
[run: train] ──uses────────────────────┘
             ──produces──>  (artifact: trained-model, model)
                                        │
[run: evaluate] ──uses─────────────────┘   ──logs──> metrics
```

Alternating run-node / artifact-node is EXACTLY how the W&B Graph view draws
it — internalize this picture and section 8 is free.

[back to top](#table-of-contents)

## 3. stage 1 — log a dataset, line by line

```python
import wandb
import torch
from torchvision import datasets, transforms

def load(train_size=50_000):
    """Load MNIST, split train into train/val, return three datasets."""
    train = datasets.MNIST("./", train=True, download=True,
                           transform=transforms.ToTensor())
    test  = datasets.MNIST("./", train=False, download=True,
                           transform=transforms.ToTensor())
    train, val = torch.utils.data.random_split(
        train, [train_size, len(train) - train_size])
    return {"training": train, "validation": val, "test": test}

def load_and_log():
    with wandb.init(project="artifacts-demo",
                    job_type="load-data") as run:          # 1

        datasets_dict = load()

        raw_data = wandb.Artifact(
            "mnist-raw", type="dataset",                    # 2
            description="Raw MNIST, split into train/val/test",
            metadata={"source": "torchvision.datasets.MNIST",# 3
                      "sizes": {k: len(v) for k, v in datasets_dict.items()}})

        for name, data in datasets_dict.items():
            with raw_data.new_file(name + ".pt", mode="wb") as f:   # 4
                x, y = data.dataset.data, data.dataset.targets      # (simplified)
                torch.save((x, y), f)

        run.log_artifact(raw_data)                          # 5

load_and_log()
```

The five graded beats:
1. `job_type="load-data"` — labels this run's ROLE in the pipeline; the graph
   view and filters group by it.
2. Artifact created with name + type (+human description).
3. **metadata** — an arbitrary dict riding along (sizes, source, params).
   Queryable later; the difference between a blob and a documented dataset.
4. `artifact.new_file(name)` — opens a file handle INSIDE the artifact's
   staging area; whatever you write becomes part of the package. (vs
   `add_file(path)` which grabs an already-existing file — section 5.)
5. `run.log_artifact(art)` — checksum, upload (dedup'd), register as the next
   version. Producer side complete.

[back to top](#table-of-contents)

## 4. stage 2 — use and preprocess, line by line

```python
def preprocess(dataset, normalize=True, expand_dims=True):
    x, y = dataset.tensors if hasattr(dataset, "tensors") else dataset
    if normalize:
        x = x.type(torch.float32) / 255.0          # uint8 -> [0,1]
    if expand_dims:
        x = torch.unsqueeze(x, 1)                  # (N,28,28) -> (N,1,28,28)
    return torch.utils.data.TensorDataset(x, y)

def preprocess_and_log(steps):
    with wandb.init(project="artifacts-demo", job_type="preprocess-data") as run:

        processed_data = wandb.Artifact(
            "mnist-preprocess", type="dataset",
            description="Preprocessed MNIST",
            metadata=steps)                          # RECORD the recipe!

        raw_data_artifact = run.use_artifact("mnist-raw:latest")   # A
        raw_dataset_dir = raw_data_artifact.download()             # B

        for split in ["training", "validation", "test"]:
            raw_split = read(raw_dataset_dir, split)               # torch.load
            processed_dataset = preprocess(raw_split, **steps)
            with processed_data.new_file(split + ".pt", mode="wb") as f:
                torch.save(processed_dataset.tensors, f)

        run.log_artifact(processed_data)

preprocess_and_log(steps={"normalize": True, "expand_dims": True})
```

The consumer-side pair to memorize:
- **A. `run.use_artifact("name:alias")`** — declares the dependency. Even
  before downloading, this single call is what draws the lineage edge
  raw-data -> this-run in the graph.
- **B. `.download()`** — pulls the files to a local cache dir (returned path).
  Cached: second call on the same version downloads nothing.

And the ops gem: `metadata=steps` — the preprocessing PARAMETERS are stored ON
the output artifact. Six months later, "was v2 normalized?" is a metadata
lookup, not archaeology.

[back to top](#table-of-contents)

## 5. stage 3 — train and log the model

```python
def train_and_log(config):
    with wandb.init(project="artifacts-demo", job_type="train",
                    config=config) as run:
        config = wandb.config

        data_art = run.use_artifact("mnist-preprocess:latest")     # consume data
        data_dir = data_art.download()
        train_loader, val_loader = make_loaders(data_dir, config.batch_size)

        model = ConvNet(config).to(device)          # small CNN in the notebook
        train(model, train_loader, val_loader, config)   # standard loop,
                                                         # wandb.log inside

        torch.save(model.state_dict(), "trained_model.pth")

        model_artifact = wandb.Artifact(
            "trained-model", type="model",
            description="ConvNet trained on preprocessed MNIST",
            metadata=dict(config))                   # hyperparams ON the model

        model_artifact.add_file("trained_model.pth")     # attach existing file
        run.log_artifact(model_artifact)
```

- **`add_file(path)`** vs new_file: the weights file already exists on disk
  (torch.save wrote it), so we ATTACH it. new_file = write-into;
  add_file = point-at. `add_dir(path)` packages a whole folder.
- One run both USES (data) and PRODUCES (model) — most real pipeline stages do;
  the graph gets an in-edge and an out-edge.
- `metadata=dict(config)` — the model artifact carries its own hyperparameters.
  Model + config + data-lineage = the full birth certificate.

[back to top](#table-of-contents)

## 6. stage 4 — evaluate a logged model

```python
def evaluate_and_log():
    with wandb.init(project="artifacts-demo", job_type="evaluate") as run:

        model_art = run.use_artifact("trained-model:latest")
        model_dir = model_art.download()
        model = ConvNet(model_art.metadata)               # rebuild from metadata!
        model.load_state_dict(
            torch.load(os.path.join(model_dir, "trained_model.pth")))
        model.to(device).eval()

        data_art = run.use_artifact("mnist-preprocess:latest")
        test_loader = make_test_loader(data_art.download())

        loss, accuracy, hardest = evaluate(model, test_loader)
        wandb.log({"test_loss": loss, "test_accuracy": accuracy})

        # the notebook's flourish: log the most-wrong images for eyeballing
        wandb.log({"hardest examples":
                   [wandb.Image(x, caption=f"pred:{p} true:{t}")
                    for x, p, t in hardest]})
```

Two exam-worthy touches:
- Rebuilding the model FROM the artifact's metadata — the stored config tells
  you the architecture hyperparameters; no hard-coding.
- `wandb.Image(tensor, caption=...)` — rich media logging; the "hardest
  examples" gallery (highest-loss test images) is a debugging pattern worth
  naming: it shows WHAT the model fails on, not just how much.

[back to top](#table-of-contents)

## 7. versioning mechanics — hashes, aliases, metadata

How versions actually work — the part theory questions target:

- On `log_artifact`, W&B computes a **content checksum** (per file + overall).
  - Content CHANGED since the last version -> new version (v0 -> v1).
  - Content IDENTICAL -> **no new version, no re-upload** (dedup). Logging the
    same dataset daily costs nothing until it actually changes.
- **Aliases** are movable name tags on versions:
  - `latest` — auto-moves to the newest version.
  - `v0, v1, ...` — permanent.
  - Custom: `best`, `prod` — YOU move them:
    `run.log_artifact(art, aliases=["latest", "prod"])`.
  - Consumers pin what they need: `use_artifact("trained-model:prod")` for
    stability, `:latest` for freshness — deployment policy expressed as a string.
- **Metadata** — dict on the artifact; **description** — human text; both
  visible/queryable in the UI and API.

```
trained-model:  v0 ── v1 ── v2 ── v3
                       ▲            ▲
                     [prod]      [latest]     <- aliases point at versions
```

[back to top](#table-of-contents)

## 8. the graph view — lineage

On any artifact's page -> Graph/Lineage tab:

- Nodes alternate: **runs** (rectangles, labeled by job_type) and
  **artifacts** (rounded, labeled name:version).
- Edges: run -> artifact it LOGGED; artifact -> run that USED it.
- The picture from section 2, drawn for you automatically — because every
  `use_artifact` and `log_artifact` call registered an edge.
- **Exploded view** toggle: collapses/expands versions — see either the tidy
  pipeline shape or every concrete version's history.

Why it matters (the answer to "what problem does lineage solve?"): impact
analysis and root-cause both become graph walks — "dataset v2 was bad; which
models descend from it?" downstream walk; "prod model misbehaves; what data
made it?" upstream walk.

[back to top](#table-of-contents)

## 9. api mini-reference — every call

```
PRODUCER
  art = wandb.Artifact(name, type, description=..., metadata={...})
  art.new_file(fname)            # write a new file directly into the artifact
  art.add_file(path)             # attach an existing file
  art.add_dir(path)              # attach a whole directory
  run.log_artifact(art, aliases=["latest", ...])   # upload + version

CONSUMER
  art = run.use_artifact("name:alias")   # declare dependency (lineage edge!)
  path = art.download()                  # cached local copy; returns dir
  art.metadata / art.description / art.version

RUN CONTEXT
  wandb.init(project=..., job_type="load-data|preprocess|train|evaluate")

RICH LOGGING
  wandb.Image(img, caption=...)   inside wandb.log({...})
  wandb.save(filepath)            # loose-file attach to a run (pre-artifact way)
```

[back to top](#table-of-contents)

## 10. mnemonics

- **"Git for heavy things"** — artifacts = versioned data/models with lineage.
- **"NAL / UD"** — producer: New-file/Add-file, Log_artifact; consumer:
  Use_artifact, Download.
- **"new writes in, add points at"** — new_file vs add_file.
- **"Hash decides the version"** — same bytes, no new version; changed bytes, +1.
- **"latest moves, v0 stays, prod is yours to point."** — alias behavior.
- **"Runs are rectangles, artifacts are rounded"** — reading the graph.
- **"Metadata is the birth certificate"** — configs/steps ride on artifacts.

[back to top](#table-of-contents)

## 11. cheatsheet

```
CREATE      wandb.Artifact("mnist-raw", type="dataset",
                           description=..., metadata={...})
FILL        with art.new_file("train.pt", mode="wb") as f: torch.save(obj, f)
            art.add_file("model.pth") | art.add_dir("data/")
PUBLISH     run.log_artifact(art, aliases=["latest","prod"])
CONSUME     art = run.use_artifact("mnist-preprocess:latest")
            dir = art.download()      # cached
JOB TYPES   wandb.init(job_type="load-data"|"preprocess-data"|"train"|"evaluate")
PIPELINE    load->raw(ds)  preprocess->processed(ds)  train->model  evaluate->metrics
VERSIONS    content-hash dedup; v0,v1 permanent; latest auto; custom movable
LINEAGE     use_artifact + log_artifact calls draw the run/artifact DAG
MEDIA       wandb.log({"hardest": [wandb.Image(x, caption=...)]})
MODEL META  rebuild architecture from art.metadata; weights from downloaded file
```

[back to top](#table-of-contents)

## 12. exam hacks and trap watch

1. use_artifact draws the lineage edge even without download() — the
   declaration IS the dependency record.
2. Logging identical content does NOT create a version — "how many versions
   after logging the same file 3x?" -> one (v0).
3. `latest` is a moving alias — pin `v2` or `prod` for reproducible pipelines;
   `:latest` in prod code is the anti-pattern to call out.
4. new_file vs add_file — write-into vs point-at; add_dir for folders.
5. job_type is on wandb.init (the RUN), not on the Artifact — placement MCQ.
6. Artifact name scopes the version chain; changing the name starts a NEW
   artifact at v0, not v(n+1).
7. download() returns a DIRECTORY path — join the filename yourself.
8. Metadata on the OUTPUT artifact should record the transform params
   (the notebook passes `steps` as metadata) — the documented-pipeline habit.
9. wandb.save() (module 7) = loose file on a run; Artifact = versioned,
   typed, lineage-tracked — know which is which.
10. Graph reading: rectangles=runs (job_type), rounded=artifacts
    (name:version); arrows point produce->artifact->consume.

[back to top](#table-of-contents)

---
[Hub](DLOPS_EXAM_00_Hub.md) | [Prev: Official Sweeps](DLOPS_08_WandB_Sweeps_Official.md) | [Next: Distributed](DLOPS_10_Distributed_Training.md)
