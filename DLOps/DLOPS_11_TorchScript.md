# DLOPS_11 — TorchScript (trace vs script, jit.save/load, C++ deployment)

[Hub](DLOPS_EXAM_00_Hub.md) | [Prev: Distributed](DLOPS_10_Distributed_Training.md) | [Next: ONNX](DLOPS_12_ONNX.md)

> Story: your trained model is a brilliant chef who only speaks Python — slow
> to serve, impossible to hire into a C++ restaurant or a phone. TorchScript
> writes the chef's technique down as a **frozen, self-contained recipe card**
> (code + weights in one file) that any kitchen can execute without Python.
> Two ways to write the card: **trace** (watch one cooking session and note
> what happened) or **script** (translate the actual recipe text, keeping all
> the "if the sauce splits, then..." branches).

## table of contents
- [1. why torchscript — the deployment gap](#1-why-torchscript--the-deployment-gap)
- [2. the eager model — mycell](#2-the-eager-model--mycell)
- [3. tracing — jit.trace, line by line](#3-tracing--jittrace-line-by-line)
- [4. graph and code — inspecting the ir](#4-graph-and-code--inspecting-the-ir)
- [5. where tracing fails — control flow](#5-where-tracing-fails--control-flow)
- [6. scripting — jit.script](#6-scripting--jitscript)
- [7. mixing trace and script](#7-mixing-trace-and-script)
- [8. save, load, and the c++ story](#8-save-load-and-the-c-story)
- [9. trace vs script — the decision table](#9-trace-vs-script--the-decision-table)
- [10. mnemonics](#10-mnemonics)
- [11. cheatsheet](#11-cheatsheet)
- [12. exam hacks and trap watch](#12-exam-hacks-and-trap-watch)

---

## 1. why torchscript — the deployment gap

Eager PyTorch (what you've written all course) = normal Python executing op
by op. Perfect for research; problematic for serving:

1. **Needs a Python interpreter** — heavy for mobile/embedded, GIL-limited
   for high-throughput servers.
2. **Code and weights are separate** — state_dict needs the class definition;
   shipping = shipping your source.
3. **No whole-graph view** — the runtime can't fuse/optimize ops it hasn't
   seen yet.

TorchScript answers all three: a **statically-analyzable subset of Python**
compiled to an intermediate representation (IR), serialized WITH the weights
into one `.pt` archive, runnable from **libtorch (C++)**, mobile, or Python —
and optimizable (op fusion, dead code elimination) because the whole graph is
known ahead of time.

```
eager:   python code + state_dict  --needs-->  python + your class defs
script:  one model.pt (code IN the file) --needs--> any torchscript runtime
```

[back to top](#table-of-contents)

## 2. the eager model — mycell

The notebook's running example — a tiny RNN-ish cell:

```python
import torch

class MyCell(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, x, h):
        new_h = torch.tanh(self.linear(x) + h)   # mix input + hidden state
        return new_h, new_h                      # (output, new hidden)

my_cell = MyCell()
x, h = torch.rand(3, 4), torch.rand(3, 4)
print(my_cell(x, h))
```

Notes:
- Two inputs (x and hidden state h), two outputs — chosen so you learn that
  trace handles multi-input/multi-output via a TUPLE of example inputs.
- tanh keeps hidden values in (-1,1) — the classic recurrent squash
  (module 2's activation zoo, applied).

[back to top](#table-of-contents)

## 3. tracing — jit.trace, line by line

```python
traced_cell = torch.jit.trace(my_cell, (x, h))   # run once, RECORD the ops
print(traced_cell)          # TracedModule wrapper
traced_cell(x, h)           # call it exactly like the original
```

What trace actually does — the mechanism (write this as the definition):
1. Runs `my_cell(x, h)` with your example inputs.
2. Records every tensor OPERATION that executes into a graph.
3. Returns a `ScriptModule` whose forward replays that recorded graph.

Consequences that follow directly from "records what EXECUTED":
- Anything that isn't a tensor op is invisible: `print()` disappears, Python
  side effects disappear.
- Data-dependent branches are BAKED to whatever path the example input took
  (section 5 — the big one).
- Input SHAPES can be somewhat flexible, but structure (number of args,
  dtypes) is fixed by the example.
- The example inputs tuple `(x, h)` — multiple args passed as a tuple.

Sanity habit: compare outputs before trusting a trace:

```python
assert torch.allclose(my_cell(x, h)[0], traced_cell(x, h)[0])
```

[back to top](#table-of-contents)

## 4. graph and code — inspecting the ir

Two views of the compiled thing:

```python
print(traced_cell.graph)
# %x : Float(3, 4), %h : Float(3, 4) ...
# %linear : ... = prim::GetAttr[name="linear"](%self)
# %8 : Float(3, 4) = aten::add(%6, %h, %alpha)
# %9 : Float(3, 4) = aten::tanh(%8)  ...
```

- `.graph` — the raw IR: SSA-form nodes (`aten::add`, `aten::tanh`), typed
  values, low-level and verbose. `aten` = PyTorch's C++ op library namespace.

```python
print(traced_cell.code)
# def forward(self, x: Tensor, h: Tensor):
#     linear = self.linear
#     _0 = torch.tanh(torch.add((linear).forward(x), h))
#     return (_0, _0)
```

- `.code` — a Python-syntax rendering of the same graph: human-readable,
  and the fastest way to SEE what trace captured (or dropped).

Rule of thumb: **debug traces by reading .code** — a missing if-branch is
obvious there.

[back to top](#table-of-contents)

## 5. where tracing fails — control flow

The notebook's deliberate failure — a submodule with a data-dependent branch:

```python
class MyDecisionGate(torch.nn.Module):
    def forward(self, x):
        if x.sum() > 0:          # decision depends on the DATA
            return x
        else:
            return -x

class MyCell(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dg = MyDecisionGate()
        self.linear = torch.nn.Linear(4, 4)
    def forward(self, x, h):
        new_h = torch.tanh(self.dg(self.linear(x)) + h)
        return new_h, new_h

traced_cell = torch.jit.trace(MyCell(), (x, h))
# TracerWarning: Converting a tensor to a Python boolean might cause the
# trace to be incorrect...
print(traced_cell.dg.code)
# def forward(self, x):
#     return x          <- the else branch is GONE
```

Read the disaster:
- During tracing, `x.sum() > 0` evaluated to (say) True — so ONLY the
  `return x` path was recorded. The trace will return x FOREVER, even for
  inputs where the sum is negative.
- PyTorch does warn (**TracerWarning**) — but the trace still "works,"
  silently wrong. Warning + silently-wrong = the exact phrase to use.
- Same failure mode for loops with data-dependent lengths.

**Mnemonic — "Trace is a tourist: it photographs the road it took, not the map."**

[back to top](#table-of-contents)

## 6. scripting — jit.script

The fix — compile the SOURCE, not a recording:

```python
scripted_gate = torch.jit.script(MyDecisionGate())
scripted_cell = torch.jit.script(MyCell())

print(scripted_gate.code)
# def forward(self, x):
#     if bool(torch.gt(torch.sum(x), 0)):
#         _0 = x
#     else:
#         _0 = torch.neg(x)
#     return _0                <- BOTH branches preserved!
```

How script works: a compiler parses your forward's Python source (the
TorchScript language subset), builds the IR directly — if/else, while loops,
list ops all survive, because they're compiled, not observed.

The catch: TorchScript is a SUBSET of Python. Heavy dynamic tricks
(arbitrary objects, some libraries, fancy metaprogramming) won't compile —
you get a compile error (loud, at export time). Loud failure > trace's
silent wrongness: an underrated advantage worth stating.

[back to top](#table-of-contents)

## 7. mixing trace and script

Both produce ScriptModules — so they COMPOSE. The notebook shows both directions:

```python
# scripted submodule INSIDE a traced module:
class MyCell(torch.nn.Module):
    def __init__(self, dg):
        super().__init__()
        self.dg = dg                       # pass in scripted gate
        self.linear = torch.nn.Linear(4, 4)
    def forward(self, x, h):
        return torch.tanh(self.dg(self.linear(x)) + h)

my_cell = MyCell(torch.jit.script(MyDecisionGate()))
traced = torch.jit.trace(my_cell, (x, h))     # trace the wrapper,
                                              # script'd gate keeps its branches

# traced module INSIDE a scripted module:
class WrapRNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.loop = torch.jit.trace(MyRNNLoop(), (torch.rand(10, 3, 4)))
    def forward(self, xs):
        y, h = self.loop(xs)
        return torch.relu(y)

scripted = torch.jit.script(WrapRNN())
```

Why mix (the practical answer): script the parts with control flow, trace the
straight-line parts (tracing tolerates more Python weirdness in code that
doesn't branch). Real deployments mix freely.

[back to top](#table-of-contents)

## 8. save, load, and the c++ story

```python
traced.save("wrapped_rnn.pt")             # ONE file: code + parameters
loaded = torch.jit.load("wrapped_rnn.pt") # no class definition needed!
print(loaded.code)                        # the code came along
```

- Contrast with state_dict loading (module 2's C-L-D-E ritual): jit.load
  needs NO Python class — the architecture is IN the archive. That single
  fact is the deployment win; make it the centerpiece of any "compare saving
  methods" answer.
- The notebook's closing point — loading in **C++** with libtorch:

```cpp
#include <torch/script.h>
torch::jit::script::Module module = torch::jit::load("wrapped_rnn.pt");
std::vector<torch::jit::IValue> inputs;
inputs.push_back(torch::rand({10, 3, 4}));
at::Tensor output = module.forward(inputs).toTensor();
```

- Same file also targets mobile (PyTorch Mobile / ExecuTorch lineage) —
  "train in Python, serve anywhere TorchScript runs."

```
train (python, eager) -> jit.script/trace -> model.pt
                                             |-- python: torch.jit.load
                                             |-- C++:    torch::jit::load
                                             +-- mobile runtimes
```

[back to top](#table-of-contents)

## 9. trace vs script — the decision table

| | torch.jit.trace | torch.jit.script |
|---|---|---|
| input | model + EXAMPLE inputs | model (source code) |
| mechanism | record one execution | compile the source |
| control flow | frozen to traced path (silent + warning) | fully preserved |
| python support | any python DURING trace (ops recorded) | TorchScript subset only |
| failure mode | silent wrong results | loud compile error |
| best for | straight-line nets (CNN forward) | branches, loops, RNN logic |

Decision rule (the one-liner): **"Does forward() branch or loop on DATA?
Yes -> script. No -> trace is fine (and easier)."**

[back to top](#table-of-contents)

## 10. mnemonics

- **"TS = Trace Simple, Script Smart."** — the choice rule.
- **"Trace is a tourist: photographs the road taken, not the map."**
- **"Script reads the recipe; trace watches one dinner."**
- **"Warning + silently wrong"** — trace's control-flow failure signature.
- **".code to check, .graph to geek"** — the two inspection views.
- **"One file, no class"** — jit.save packs code + weights; jit.load needs nothing else.
- **"Loud beats silent"** — script's compile errors vs trace's quiet bugs.

[back to top](#table-of-contents)

## 11. cheatsheet

```
TRACE       traced = torch.jit.trace(model, (ex_x, ex_h))   # tuple for multi-arg
SCRIPT      scripted = torch.jit.script(model)
INSPECT     m.graph (raw IR, aten:: ops) | m.code (python-like source)
VERIFY      torch.allclose(eager_out, traced_out)
MIX         script(gate) inside trace(cell) — and vice versa; both compose
SAVE        m.save("model.pt")            # code + params, ONE archive
LOAD (py)   m = torch.jit.load("model.pt")     # no class definition needed
LOAD (c++)  torch::jit::load("model.pt")  via  #include <torch/script.h>
CHOOSE      data-dependent if/loop -> script ; plain feedforward -> trace
FAIL MODES  trace: TracerWarning + frozen branch (silent)
            script: compile error on unsupported python (loud)
```

[back to top](#table-of-contents)

## 12. exam hacks and trap watch

1. "Model has `if x.sum() > 0`" anywhere in the question -> answer is SCRIPT.
2. Trace with control flow doesn't crash — it warns and freezes one branch;
   "runs fine but wrong for half the inputs" is the expected phrasing.
3. Multi-input models trace with a TUPLE of example inputs: `trace(m, (x, h))`.
4. jit.load needs NO class definition — the discriminator vs state_dict loading.
5. `.code` shows the else-branch missing after a bad trace — the debugging move.
6. `aten::` in .graph = PyTorch's native op library — name-the-namespace freebie.
7. print() inside forward vanishes from a trace (not a tensor op).
8. TorchScript is a Python SUBSET — script can reject valid Python (loudly).
9. Verify traces with torch.allclose on eager vs traced outputs — method marks.
10. Whole-graph availability enables optimization (op fusion) — the "why is it
    faster" systems answer beyond just "no Python."

[back to top](#table-of-contents)

---
[Hub](DLOPS_EXAM_00_Hub.md) | [Prev: Distributed](DLOPS_10_Distributed_Training.md) | [Next: ONNX](DLOPS_12_ONNX.md)
