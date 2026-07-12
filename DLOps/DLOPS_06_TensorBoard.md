# DLOPS_06 — TensorBoard (SummaryWriter, scalars, graphs, PR curves, hparams)

[Hub](DLOPS_EXAM_00_Hub.md) | [Prev: Custom Datasets](DLOPS_05_Custom_Datasets_Training.md) | [Next: W&B Sweeps](DLOPS_07_WandB_Sweeps_Course.md)

> Story: training without tracking is cooking without tasting — you only find
> out at the end that it's ruined. TensorBoard is your tasting diary: every
> epoch you jot loss and accuracy into a log folder with a **SummaryWriter**,
> and a dashboard turns your notes into live charts. Then the real DLOps move:
> run a GRID of experiments, each with its own diary, and let the dashboard
> tell you which recipe wins.

> **Cross-repo reference (TS-01):** treat hparams/PR-curve tracking here as the
> tooling layer on top of TS-01 →
> [Parameter-Estimations-Guide](https://github.com/rpaut03l/TS-01/tree/main/ML/Parameter-Estimations-Guide) |
> [ml_master_gap_index.md](https://github.com/rpaut03l/TS-01/blob/main/ML/ml_master_gap_index.md)

## table of contents
- [1. install, launch, mental model](#1-install-launch-mental-model)
- [2. summarywriter — creation and log dirs](#2-summarywriter--creation-and-log-dirs)
- [3. add_scalar and add_scalars](#3-add_scalar-and-add_scalars)
- [4. add_graph — the model diagram](#4-add_graph--the-model-diagram)
- [5. add_pr_curve — precision-recall per class](#5-add_pr_curve--precision-recall-per-class)
- [6. add_hparams — the hyperparameter table](#6-add_hparams--the-hyperparameter-table)
- [7. upgrading train() with a writer](#7-upgrading-train-with-a-writer)
- [8. the experiment grid — the notebook's main event](#8-the-experiment-grid--the-notebooks-main-event)
- [9. picking and reloading the best model](#9-picking-and-reloading-the-best-model)
- [10. mnemonics](#10-mnemonics)
- [11. cheatsheet](#11-cheatsheet)
- [12. exam hacks and trap watch](#12-exam-hacks-and-trap-watch)

---

## 1. install, launch, mental model

```bash
pip install tensorboard          # writer ships with torch; this is the viewer
tensorboard --logdir=runs        # CLI: serves http://localhost:6006
```

```python
# Colab magic version:
%load_ext tensorboard
%tensorboard --logdir runs
```

Mental model:

```
your code --SummaryWriter--> event files on disk (runs/...)
                                     |
tensorboard --logdir=runs  reads them live --> browser dashboard
```

Two totally decoupled halves: the writer only WRITES files; the dashboard only
READS them. You can open the dashboard mid-training and watch curves grow.
This decoupling is the exam-worthy design point.

[back to top](#table-of-contents)

## 2. summarywriter — creation and log dirs

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()                    # default: runs/CURRENT_DATETIME
writer = SummaryWriter(log_dir="runs/exp1") # explicit folder
```

The notebook's key utility — timestamped, structured log dirs so experiments
NEVER overwrite each other:

```python
from datetime import datetime
import os

def create_writer(experiment_name: str, model_name: str, extra: str = None):
    timestamp = datetime.now().strftime("%Y-%m-%d")        # e.g. 2026-07-12
    parts = ["runs", timestamp, experiment_name, model_name] + ([extra] if extra else [])
    log_dir = os.path.join(*parts)
    print(f"[INFO] Created SummaryWriter, saving to: {log_dir}")
    return SummaryWriter(log_dir=log_dir)

writer = create_writer("data_10_percent", "tinyvgg", "5_epochs")
# -> runs/2026-07-12/data_10_percent/tinyvgg/5_epochs
```

Why this structure matters: the dashboard's left sidebar groups runs BY
FOLDER PATH — a disciplined naming scheme (date/experiment/model/config) makes
comparisons one checkbox away. This helper is a likely "write a function
that..." exam question.

Always finish with:

```python
writer.close()      # flushes buffers; without it the last points may be lost
```

[back to top](#table-of-contents)

## 3. add_scalar and add_scalars

The bread and butter — one number per step:

```python
writer.add_scalar(tag="Loss/train",        # slash groups charts: 'Loss' section
                  scalar_value=train_loss, # the y value
                  global_step=epoch)       # the x value
```

Multiple lines on ONE chart (comparing train vs test):

```python
writer.add_scalars(main_tag="Loss",
                   tag_scalar_dict={"train_loss": train_loss,
                                    "test_loss":  test_loss},
                   global_step=epoch)
writer.add_scalars("Accuracy",
                   {"train_acc": train_acc, "test_acc": test_acc},
                   global_step=epoch)
```

- `add_scalar` = one line per tag; `add_scalars` = a dict of lines sharing a chart.
- `global_step` is the x-axis: epoch for per-epoch logging, batch counter for
  per-batch logging.
- Tag naming with `/` creates dashboard sections: `Loss/train`, `Loss/test`
  fold under "Loss".

```
dashboard SCALARS tab:
  Loss      [ train ----\____ ; test ----\__/-- ]   <- the overfit gap, live
  Accuracy  [ train  __/----- ; test  __/--\    ]
```

[back to top](#table-of-contents)

## 4. add_graph — the model diagram

```python
writer.add_graph(model=model,
                 input_to_model=torch.randn(32, 3, 64, 64).to(device))
```

- Traces one forward pass with the dummy input and renders the computation
  graph in the GRAPHS tab — clickable blocks showing each layer and the
  tensor shapes flowing between them.
- The dummy input must be on the SAME device as the model and have a valid
  shape — the two classic errors here.
- Use: visually verify architecture and shape flow (torchinfo's graphical twin).

[back to top](#table-of-contents)

## 5. add_pr_curve — precision-recall per class

Accuracy is one number; PR curves show the precision/recall TRADE-OFF as the
decision threshold slides — per class:

```python
# collect predictions over the test set
class_probs, class_labels = [], []
model.eval()
with torch.inference_mode():
    for X, y in test_loader:
        logits = model(X.to(device))
        class_probs.append(torch.softmax(logits, dim=1))   # probabilities!
        class_labels.append(y)
probs = torch.cat(class_probs)        # (N, C)
labels = torch.cat(class_labels)      # (N,)

for i, name in enumerate(class_names):
    writer.add_pr_curve(tag=name,
                        labels=(labels == i),   # binary: is-this-class?
                        predictions=probs[:, i],# prob of this class
                        global_step=0)
```

Reading a PR curve (one-liner definitions to memorize):
- **Precision** = of everything I flagged as pizza, how much was pizza (TP/(TP+FP)).
- **Recall** = of all real pizzas, how many did I catch (TP/(TP+FN)).
- Curve hugging the top-right = great classifier for that class; a sagging
  curve exposes the weak class that overall accuracy hides.

[back to top](#table-of-contents)

## 6. add_hparams — the hyperparameter table

```python
writer.add_hparams(hparam_dict={"lr": 0.001, "batch_size": 32, "epochs": 5},
                   metric_dict={"accuracy": test_acc, "loss": test_loss})
```

- Logs a ROW into the HPARAMS tab: settings on the left, outcomes on the right.
- After many runs, the tab becomes a sortable table + parallel-coordinates
  view — "which lr rows have the best accuracy?" answered by a click.
- This is the manual ancestor of W&B sweeps (modules 7-8): here YOU run each
  combination; there an agent does.

[back to top](#table-of-contents)

## 7. upgrading train() with a writer

The module-5 trio gains one parameter — spot the only additions:

```python
def train(model, train_dataloader, test_dataloader, optimizer,
          loss_fn, epochs, writer=None):                      # NEW param
    results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    for epoch in tqdm(range(epochs)):
        tr_loss, tr_acc = train_step(model, train_dataloader, loss_fn, optimizer)
        te_loss, te_acc = test_step(model, test_dataloader, loss_fn)
        # ...append to results, print...
        if writer:                                            # NEW block
            writer.add_scalars("Loss", {"train_loss": tr_loss,
                                        "test_loss": te_loss}, epoch)
            writer.add_scalars("Accuracy", {"train_acc": tr_acc,
                                            "test_acc": te_acc}, epoch)
            writer.add_graph(model, torch.randn(32, 3, 64, 64).to(device))
            writer.close()
    return results
```

Design lesson (say it): tracking bolts ON to a good training function — the
loop logic is untouched. `writer=None` default keeps it usable without
TensorBoard. Same pattern repeats with wandb.log in module 7.

[back to top](#table-of-contents)

## 8. the experiment grid — the notebook's main event

The question: what helps a food classifier more — more data, more epochs, or a
bigger model? Answer it like an engineer: a GRID, one writer per run.

Setup — the axes:

```python
num_epochs = [5, 10]
models = ["effnetb0", "effnetb2"]              # or tinyvgg variants
train_dataloaders = {"data_10_percent": train_dl_10,
                     "data_20_percent": train_dl_20}
```

The grid loop (structure to memorize; details flex):

```python
experiment_number = 0
for dataloader_name, train_dataloader in train_dataloaders.items():
    for epochs in num_epochs:
        for model_name in models:
            experiment_number += 1
            print(f"Experiment {experiment_number}: {model_name} | "
                  f"{dataloader_name} | {epochs} epochs")
            model = create_model(model_name).to(device)          # FRESH model!
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            loss_fn = nn.CrossEntropyLoss()
            train(model, train_dataloader, test_dataloader, optimizer,
                  loss_fn, epochs,
                  writer=create_writer(dataloader_name, model_name,
                                       f"{epochs}_epochs"))
            save_model(model, "models",
                       f"{model_name}_{dataloader_name}_{epochs}e.pth")
```

2 x 2 x 2 = 8 experiments, 8 log folders, 8 checkpoints. The save_model utility:

```python
from pathlib import Path
def save_model(model, target_dir, model_name):
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)
    assert model_name.endswith((".pth", ".pt"))
    save_path = target_dir_path / model_name
    torch.save(model.state_dict(), save_path)
```

Two experiment-design commandments (the notebook's section 7.1 — quote them):
1. **Change one variable at a time**; the grid's structure enforces it.
2. **Start small** — 10/20% data, 5/10 epochs — scale only what wins.
A FRESH model per run (not reusing a trained one) keeps runs comparable —
the subtle bug the loop structure prevents.

[back to top](#table-of-contents)

## 9. picking and reloading the best model

- In TensorBoard: tick all 8 runs, sort the Accuracy chart / HPARAMS table;
  typically "biggest model + most data + most epochs" wins — but the CURVES
  show whether 10 epochs was already flattening (cost/benefit thinking).
- Reload the champion and predict on a custom image (module 5 section 9 flow):

```python
best = create_model("effnetb2").to(device)
best.load_state_dict(torch.load("models/effnetb2_data_20_percent_10e.pth"))
best.eval()
# read_image -> /255 -> resize -> unsqueeze(0) -> inference_mode -> softmax
```

Ops moral: because every run was TRACKED and SAVED with a naming convention,
"which model is in prod and why" has a paper trail — the whole point of DLOps.

[back to top](#table-of-contents)

## 10. mnemonics

- **SGPH — "Scalars, Graph, PR-curve, Hparams"** — the four add_* calls of class.
- **"Writer writes, dashboard reads"** — decoupled halves via event files.
- **"Date/Experiment/Model/Extra"** — the create_writer folder recipe.
- **"Close the diary"** — writer.close() flushes; forget it, lose the last page.
- **"One knob per run, fresh model per run"** — grid discipline.
- **"Precision = trust my flags; Recall = catch them all."**

[back to top](#table-of-contents)

## 11. cheatsheet

```
INSTALL/RUN  pip install tensorboard ; tensorboard --logdir=runs (port 6006)
COLAB        %load_ext tensorboard ; %tensorboard --logdir runs
WRITER       SummaryWriter(log_dir=...) ; create_writer(exp, model, extra)
SCALAR       writer.add_scalar("Loss/train", value, epoch)
SCALARS      writer.add_scalars("Loss", {"train":a,"test":b}, epoch)
GRAPH        writer.add_graph(model, torch.randn(32,3,64,64).to(device))
PR CURVE     add_pr_curve(name, labels==i, probs[:,i], step)   softmax probs!
HPARAMS      add_hparams({"lr":..., "bs":...}, {"acc":..., "loss":...})
CLOSE        writer.close()   ALWAYS
GRID         nested loops (data x epochs x model) -> fresh model + own writer
             + save_model(state_dict) per run
PICK BEST    dashboard compare -> load_state_dict -> eval -> predict
```

[back to top](#table-of-contents)

## 12. exam hacks and trap watch

1. add_graph's dummy input on CPU while model on GPU -> device-mismatch error.
2. Reusing one log_dir for two runs -> curves overlap into spaghetti; the
   create_writer scheme is the fix (know WHY, not just the code).
3. PR curves need PROBABILITIES (post-softmax), not logits.
4. `labels == i` builds the one-vs-rest binary labels — per-class curves.
5. add_scalars takes a DICT; add_scalar a single value — s counts.
6. global_step forgotten -> everything plots at x=0, a single dot.
7. Fresh model per grid run — reusing the trained one invalidates comparison.
8. HPARAMS tab needs BOTH dicts (settings + metrics) to build the table.
9. Event files live under log_dir; --logdir must point at the PARENT `runs/`.
10. "TensorBoard vs W&B?" — local files + free vs cloud + collab + sweeps
    agents; both log scalars — a compare-question freebie.

[back to top](#table-of-contents)

---
[Hub](DLOPS_EXAM_00_Hub.md) | [Prev: Custom Datasets](DLOPS_05_Custom_Datasets_Training.md) | [Next: W&B Sweeps](DLOPS_07_WandB_Sweeps_Course.md)
