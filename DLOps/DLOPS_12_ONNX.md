# DLOPS_12 — ONNX (torch.onnx.export, checker, onnxruntime InferenceSession)

[Hub](DLOPS_EXAM_00_Hub.md) | [Prev: TorchScript](DLOPS_11_TorchScript.md)

> Story: TorchScript freed your model from Python — but it still speaks only
> PyTorch. **ONNX (Open Neural Network Exchange)** is the universal passport:
> export once into a standard graph format that TensorFlow-land, mobile chips,
> NVIDIA TensorRT, Intel OpenVINO, and the lightweight **ONNX Runtime** can all
> execute. Train wherever you like; serve on whatever hardware the ops team
> hands you. This covers class notebooks 12 (intro) and 13 (export walkthrough)
> as one module.

## table of contents
- [1. what onnx is — format, ops, ecosystem](#1-what-onnx-is--format-ops-ecosystem)
- [2. when onnx vs torchscript](#2-when-onnx-vs-torchscript)
- [3. setup and the model to export](#3-setup-and-the-model-to-export)
- [4. torch.onnx.export — every argument](#4-torchonnxexport--every-argument)
- [5. dynamic axes — variable batch sizes](#5-dynamic-axes--variable-batch-sizes)
- [6. validating — onnx.load and the checker](#6-validating--onnxload-and-the-checker)
- [7. netron — seeing the graph](#7-netron--seeing-the-graph)
- [8. onnxruntime — inference, line by line](#8-onnxruntime--inference-line-by-line)
- [9. verifying pytorch vs onnx outputs](#9-verifying-pytorch-vs-onnx-outputs)
- [10. the full deployment decision tree](#10-the-full-deployment-decision-tree)
- [11. mnemonics](#11-mnemonics)
- [12. cheatsheet](#12-cheatsheet)
- [13. exam hacks and trap watch](#13-exam-hacks-and-trap-watch)

---

## 1. what onnx is — format, ops, ecosystem

Definition to write: **ONNX is an open, framework-neutral file format for ML
models — a serialized computation graph (protobuf) whose nodes come from a
standardized operator set (opset), plus the trained weights.**

The three pieces:
1. **Graph** — nodes (Conv, Relu, Gemm, ...), edges (tensors), inputs/outputs.
2. **Opset** — the versioned dictionary of allowed operators. Exporters
   target an `opset_version`; runtimes declare which opsets they support.
   Version mismatches are THE classic ONNX headache — know the word "opset."
3. **Weights** — initializer tensors baked into the same file.

Ecosystem map:

```
 PRODUCERS (train here)          model.onnx           CONSUMERS (serve here)
 PyTorch  ──torch.onnx.export──►   ┌──────────┐  ──►  ONNX Runtime (CPU/GPU, tiny)
 TensorFlow/Keras ──tf2onnx────►   │ graph    │  ──►  TensorRT   (NVIDIA, fastest)
 sklearn ──skl2onnx────────────►   │ +opset   │  ──►  OpenVINO   (Intel)
                                   │ +weights │ ──►  CoreML/mobile, browsers (wasm)
                                   └──────────┘
```

The decoupling sentence for theory answers: **ONNX separates the TRAINING
framework from the SERVING runtime — teams can standardize deployment on one
runtime while researchers use any framework.**

[back to top](#table-of-contents)

## 2. when onnx vs torchscript

| | TorchScript | ONNX |
|---|---|---|
| ecosystem | PyTorch-only runtimes | framework-neutral, many runtimes |
| fidelity | full TorchScript language (control flow via script) | graph of standard ops; exotic ops may not map |
| hardware accel | libtorch targets | TensorRT/OpenVINO/NPU toolchains eat ONNX |
| typical use | C++/mobile serving inside PyTorch world | cross-framework, vendor-optimized serving |

Rule of thumb: **staying in the PyTorch family -> TorchScript; leaving it
(or chasing vendor-optimized inference) -> ONNX.** Both start from the same
place: a trained nn.Module.

[back to top](#table-of-contents)

## 3. setup and the model to export

```bash
pip install onnx onnxruntime        # onnx = format tools ; onnxruntime = engine
```

Two packages, two jobs (a subtle MCQ): `onnx` loads/checks/edits model files;
`onnxruntime` EXECUTES them. You can serve with onnxruntime alone.

The model — any trained module works; here the course-flavored example:

```python
import torch
import torch.nn as nn

class Net(nn.Module):                    # the module-3 CNN or any classifier
    ...                                  # (architecture as trained)

model = Net()
model.load_state_dict(torch.load("cifar_net.pth"))
model.eval()                             # CRITICAL before export (dropout/bn!)
```

`model.eval()` before export is graded: export runs a forward pass — with
dropout active you'd bake training-mode randomness into the exported graph's
behavior expectations. Eval first, always.

[back to top](#table-of-contents)

## 4. torch.onnx.export — every argument

```python
dummy_input = torch.randn(1, 3, 32, 32)      # example input: defines shapes

torch.onnx.export(
    model,                       # the eval-mode module
    dummy_input,                 # example input (tuple for multi-input)
    "model.onnx",                # output file path
    export_params=True,          # bake trained weights INTO the file (default)
    opset_version=17,            # operator-set version to target
    do_constant_folding=True,    # pre-compute constant subgraphs (optimization)
    input_names=["input"],       # name the graph inputs...
    output_names=["output"],     # ...and outputs (used as dict keys at runtime!)
    dynamic_axes={"input":  {0: "batch_size"},     # section 5
                  "output": {0: "batch_size"}},
)
```

Argument-by-argument, with the WHY:
- **model + dummy_input** — export TRACES a forward pass (TorchScript trace
  under the hood!) to capture the graph. Hence the dummy: it defines input
  shape/dtype. And hence trace's limitation inherited: **data-dependent
  control flow won't survive a default export** — the module-11 lesson
  applies here too.
- **export_params=True** — weights become initializers inside the file; the
  .onnx is self-contained (like jit.save, unlike state_dict).
- **opset_version** — which operator dictionary to emit. Newer opsets support
  more ops; your RUNTIME must support the chosen opset. If export fails with
  "operator not supported," trying a different opset_version is the first fix.
- **do_constant_folding** — evaluates parts of the graph that don't depend on
  input at export time (e.g. reshaping constants) — smaller, faster graph.
- **input_names / output_names** — without them you get "0", "1"...; with
  them, the runtime feed dict reads like English: `{"input": x}`.

[back to top](#table-of-contents)

## 5. dynamic axes — variable batch sizes

The trap dynamic_axes exists to fix: the dummy had batch size 1, so by
default the graph HARD-CODES batch=1 — feeding 32 images later fails with a
shape error.

```python
dynamic_axes={"input":  {0: "batch_size"},
              "output": {0: "batch_size"}}
```

- Reads as: for the tensor named "input", dimension 0 is not fixed — it's a
  symbolic size called "batch_size". Same for output.
- Now the same .onnx serves batch 1 (online inference) and batch 256
  (offline scoring) — one artifact, both serving modes. That sentence is the
  exam-answer version of "why dynamic axes."
- Any dim can be dynamic (variable-length sequences: `{0: "batch", 1: "seq"}`).

```
without dynamic_axes:  input fixed [1, 3, 32, 32]   batch 32 -> ERROR
with    dynamic_axes:  input [batch_size, 3, 32, 32]  batch anything -> OK
```

[back to top](#table-of-contents)

## 6. validating — onnx.load and the checker

Never ship an unchecked export:

```python
import onnx

onnx_model = onnx.load("model.onnx")         # parse the protobuf
onnx.checker.check_model(onnx_model)         # structural validation
print("Model checked OK")

print(onnx.helper.printable_graph(onnx_model.graph))   # text dump of the graph
```

- `check_model` verifies the graph is well-formed: nodes reference valid
  tensors, types are consistent, opset declarations are sane. It raises on
  problems, returns silently on success.
- What it does NOT check (nuance = marks): numerical equivalence with the
  original PyTorch model. That's section 9's job.
- `printable_graph` — quick text view: inputs, initializers, node list —
  the CLI-grade inspection before reaching for Netron.

[back to top](#table-of-contents)

## 7. netron — seeing the graph

**Netron** (netron.app, or `pip install netron`) — the standard visualizer:
open model.onnx and see the graph as boxes and arrows — every Conv/Gemm/Relu
node clickable with its attributes (kernel size, strides) and weight shapes.

Uses worth listing: verify the architecture exported as intended, confirm
input/output names and dynamic dims (they show as symbolic), and inspect
third-party .onnx files you didn't train. One sentence in a practical answer
("verified structure in Netron") signals real workflow.

[back to top](#table-of-contents)

## 8. onnxruntime — inference, line by line

```python
import onnxruntime as ort
import numpy as np

# 1. create the session — loads + optimizes the graph once
ort_session = ort.InferenceSession(
    "model.onnx",
    providers=["CPUExecutionProvider"],      # or CUDAExecutionProvider, ...
)

# 2. prepare input — NUMPY, not torch!
x = torch.randn(4, 3, 32, 32)                # batch of 4 (dynamic axes at work)
ort_inputs = {"input": x.numpy()}            # key = input_names from export

# 3. run
ort_outputs = ort_session.run(
    None,                # output names to fetch; None = all outputs
    ort_inputs,          # {input_name: numpy_array}
)

logits = ort_outputs[0]                      # list of numpy arrays, in order
pred = np.argmax(logits, axis=1)
```

Line-by-line essentials:
1. **InferenceSession** parses the model, applies graph optimizations, and
   picks **execution providers** — the pluggable backends (CPU, CUDA,
   TensorRT, CoreML...). Provider list is priority-ordered. Create the
   session ONCE and reuse it (it's the expensive step) — the serving pattern.
2. **NumPy in** — onnxruntime doesn't know torch tensors. `x.numpy()` (and
   `.cpu()` first if on GPU). dtype must match export (float32).
3. **run(None, feed_dict)** — first arg selects outputs by name (None = all);
   returns a LIST of numpy arrays. **NumPy out** too.

The tagline to memorize: **"NumPy in, NumPy out — torch is not invited to
inference."**

[back to top](#table-of-contents)

## 9. verifying pytorch vs onnx outputs

The professional last step — prove the export didn't change the math:

```python
model.eval()
with torch.inference_mode():
    torch_out = model(x)                     # PyTorch reference

ort_out = ort_session.run(None, {"input": x.numpy()})[0]

np.testing.assert_allclose(
    torch_out.numpy(), ort_out,
    rtol=1e-3, atol=1e-5,                    # float tolerance, NOT equality
)
print("Exported model matches PyTorch. ✓ within tolerance")
```

- Tolerances, not `==`: different op implementations/orderings produce tiny
  float diffs; rtol/atol define "same enough." Expecting bit-exact equality
  is the naive-answer trap.
- This check + the structural checker (section 6) = the two-gate validation:
  **check_model for structure, assert_allclose for numbers.**

[back to top](#table-of-contents)

## 10. the full deployment decision tree

The course's entire deployment arc in one diagram — worth reproducing in any
"compare deployment options" long answer:

```
trained nn.Module
   |
   |-- staying in Python + PyTorch?
   |       -> torch.save(state_dict) + C-L-D-E ritual         [module 2]
   |
   |-- C++/mobile, PyTorch ecosystem?
   |       -> TorchScript: trace (no branches) / script (branches)
   |          -> m.save(.pt) -> torch::jit::load               [module 11]
   |
   +-- cross-framework / vendor-optimized runtime / NPU?
           -> torch.onnx.export(+dynamic_axes, opset)
              -> onnx.checker -> Netron eyeball
              -> onnxruntime InferenceSession (providers)
              -> assert_allclose vs PyTorch                    [this module]
```

**Mnemonic for the ONNX pipeline — "EC-RI-V": Export, Check, Runtime,
Infer, Verify."**

[back to top](#table-of-contents)

## 11. mnemonics

- **"ONNX = the passport; runtimes = the countries."**
- **"EC-RI-V"** — Export, Check, Runtime, Infer, Verify.
- **"opset = the dictionary edition"** — exporter writes in it, runtime must read it.
- **"eval before export"** — no dropout ghosts in the frozen graph.
- **"Dummy defines dims; dynamic_axes frees them."**
- **"NumPy in, NumPy out — torch is not invited."**
- **"checker checks structure; allclose checks numbers."**
- **"onnx reads the file, onnxruntime runs it"** — two packages, two jobs.

[back to top](#table-of-contents)

## 12. cheatsheet

```
INSTALL     pip install onnx onnxruntime
EXPORT      model.eval()
            torch.onnx.export(model, dummy, "model.onnx",
                export_params=True, opset_version=17,
                do_constant_folding=True,
                input_names=["input"], output_names=["output"],
                dynamic_axes={"input":{0:"batch"}, "output":{0:"batch"}})
CHECK       m = onnx.load(p); onnx.checker.check_model(m)
            onnx.helper.printable_graph(m.graph)
VIEW        Netron (netron.app) — nodes, attrs, shapes, symbolic dims
RUN         sess = ort.InferenceSession(p, providers=["CPUExecutionProvider"])
            out = sess.run(None, {"input": x.numpy()})[0]     # list of np arrays
VERIFY      np.testing.assert_allclose(torch_out.numpy(), out,
                                       rtol=1e-3, atol=1e-5)
GOTCHAS     float32 dtype match | session built once, reused |
            export is trace-based (control flow caveat) | opset compatibility
```

[back to top](#table-of-contents)

## 13. exam hacks and trap watch

1. Export without dynamic_axes -> batch size FROZEN to the dummy's; feeding a
   different batch errors. The #1 practical ONNX bug.
2. export runs a TRACE — module 11's control-flow warning applies to ONNX
   export too (the cross-module connection examiners love).
3. `sess.run` returns a LIST — `[0]` to get the first output; forgetting it
   and argmax-ing a list is the runtime facepalm.
4. Feed dict keys must EXACTLY match input_names from export.
5. onnx vs onnxruntime — format library vs execution engine; serving needs
   only the latter.
6. check_model validates STRUCTURE, not numerical correctness — allclose does
   the numbers. Two different gates; name both.
7. eval() before export, or dropout/batchnorm training behavior contaminates
   the traced graph.
8. Tolerances (rtol/atol) in verification — bit-exact equality across
   runtimes is not expected and not required.
9. "Operator not supported" at export -> adjust opset_version — the standard
   first remedy to name.
10. providers list is priority-ordered — ["CUDAExecutionProvider",
    "CPUExecutionProvider"] tries GPU, falls back to CPU; serving-config freebie.

[back to top](#table-of-contents)

---
[Hub](DLOPS_EXAM_00_Hub.md) | [Prev: TorchScript](DLOPS_11_TorchScript.md)
