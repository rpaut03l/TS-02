# DLOPS_EXAM_05 — Q&A Bank (answer anything asked)

[Hub](DLOPS_EXAM_00_Hub.md) | [Theory](DLOPS_EXAM_01_Theory.md) | [Numericals](DLOPS_EXAM_02_Numericals.md) | [Practice](DLOPS_EXAM_03_Practice.md) | [Deep Dive](DLOPS_EXAM_04_Notebook_Deep_Dive.md)

## table of contents
- [A. pytorch fundamentals (Q1-Q12)](#a-pytorch-fundamentals-q1-q12)
- [B. training, optimizers, regularization (Q13-Q24)](#b-training-optimizers-regularization-q13-q24)
- [C. data pipeline (Q25-Q33)](#c-data-pipeline-q25-q33)
- [D. cnn and feature extraction (Q34-Q40)](#d-cnn-and-feature-extraction-q34-q40)
- [E. tracking — tensorboard and wandb (Q41-Q50)](#e-tracking--tensorboard-and-wandb-q41-q50)
- [F. distributed and deployment (Q51-Q60)](#f-distributed-and-deployment-q51-q60)

---

## A. pytorch fundamentals (Q1-Q12)

**Q1. Tensor vs NumPy array?** Tensor adds GPU support, autograd, and integrates with nn. On CPU, `from_numpy`/`numpy()` share memory.

**Q2. What does `requires_grad=True` do?** Marks the tensor as a leaf whose operations are recorded so `backward()` can compute d(loss)/d(tensor) into `.grad`.

**Q3. `torch.no_grad()` vs `torch.inference_mode()`?** Both disable grad tracking; inference_mode is stricter and faster (also disables version counting); tensors made inside it can't later be used in autograd.

**Q4. Why call `optimizer.zero_grad()`?** `.backward()` **adds** into `.grad`; without zeroing, gradients from previous steps accumulate and updates are wrong.

**Q5. `model.eval()` vs `no_grad()` — same thing?** No. eval() flips layer behavior (Dropout off, BatchNorm uses running stats); no_grad() only stops gradient bookkeeping. Proper evaluation uses both.

**Q6. `torch.max(out,1)` returns?** A tuple (max values, argmax indices) along dim 1 — indices are the predicted classes.

**Q7. `matmul` vs `mul`?** matmul/@ = matrix product with shape rule (a,b)@(b,c)=(a,c); mul/* = elementwise with broadcasting.

**Q8. What is a state_dict?** OrderedDict mapping parameter/buffer names to tensors. Save with `torch.save(model.state_dict(), p)`; load with `model.load_state_dict(torch.load(p))`.

**Q9. Why save state_dict rather than the whole model?** Whole-model pickling ties the file to your exact class code/paths; state_dict is portable — recreate the class, load weights.

**Q10. Device-agnostic code line?** `device = "cuda" if torch.cuda.is_available() else "cpu"`; move with `.to(device)` (Mac alt: `torch.backends.mps.is_available()`).

**Q11. Why seeds?** `torch.manual_seed(42)` (+ `cuda.manual_seed`) makes random init/shuffles reproducible — core MLOps reproducibility discipline.

**Q12. What does `.item()` do?** Extracts the Python scalar from a 1-element tensor (used for logging loss/accuracy).

[back to top](#table-of-contents)

## B. training, optimizers, regularization (Q13-Q24)

**Q13. Write the 5 training-loop steps.** zero_grad -> forward -> loss -> backward -> step (ZFLBS; order of zero flexible before backward).

**Q14. Why must activation functions be non-linear?** Stacked linear layers collapse to one linear map; non-linearity lets the network model curves/complex boundaries.

**Q15. Dying ReLU and its fix?** Neurons stuck outputting 0 (negative inputs forever, zero gradient). Fix: LeakyReLU/GELU or lower lr.

**Q16. Why does CrossEntropyLoss take logits?** It internally applies log-softmax + NLL; feeding softmaxed probs double-squashes and destroys gradient scale.

**Q17. SGD momentum update equations?** `v = beta*v + g ; w = w - lr*v` — velocity smooths noisy gradients and speeds travel along consistent directions.

**Q18. Adam equations?** m and v = EMAs of g and g^2 with beta1/beta2, bias-corrected m_hat, v_hat; `w -= lr * m_hat/(sqrt(v_hat)+eps)`. Per-parameter adaptive step.

**Q19. Effect of lr too high / too low?** Too high: loss oscillates or diverges. Too low: painfully slow convergence, may stall in flat regions.

**Q20. What is dropout and when is it active?** Randomly zeroes activations with prob p during **training only** (scaled to keep expectation); acts as ensemble-like regularizer; disabled by model.eval().

**Q21. Weight decay?** L2 penalty on weights folded into optimizer (`weight_decay=`), pushing weights small -> smoother functions -> less overfitting.

**Q22. Overfitting signature on loss curves?** Train loss keeps falling, test loss flattens/rises — gap grows. Fixes: augmentation, dropout, weight decay, early stopping, more data, smaller model.

**Q23. Underfitting signature?** Both losses stay high. Fixes: bigger model, more epochs, higher lr, richer features, fewer regularizers.

**Q24. What is early stopping?** Stop training when validation loss stops improving for N epochs (patience) — keeps the best-generalizing checkpoint.

[back to top](#table-of-contents)

## C. data pipeline (Q25-Q33)

**Q25. Dataset vs DataLoader?** Dataset = indexable store (len + getitem returning one sample). DataLoader = iterator producing shuffled, batched, parallel-loaded samples.

**Q26. Three mandatory custom-Dataset methods?** `__init__`, `__len__`, `__getitem__` (ILG).

**Q27. What does ImageFolder assume?** root/class_name/image files — folder name = label; provides `.classes` and `.class_to_idx`.

**Q28. What does ToTensor do exactly?** PIL/ndarray HxWxC uint8 [0,255] -> float32 CxHxW tensor in [0,1].

**Q29. Normalize((0.5,),(0.5,)) after ToTensor gives range?** [-1,1] via (x-0.5)/0.5.

**Q30. Why shuffle train but not test?** Shuffling breaks order correlations for better gradient estimates; test needs reproducible, order-independent evaluation.

**Q31. num_workers?** Number of subprocesses pre-loading batches in parallel (often `os.cpu_count()`); 0 = load in the main process.

**Q32. What is TrivialAugmentWide?** A tuning-free augmentation policy: per image, pick one random transform at a random magnitude (bins up to 31). Used for Model 1 in class.

**Q33. Why does data augmentation reduce overfitting?** It manufactures label-preserving variety, so the model must learn invariant features instead of memorizing exact pixels.

[back to top](#table-of-contents)

## D. cnn and feature extraction (Q34-Q40)

**Q34. Why CNN over MLP for images?** Local receptive fields + weight sharing = far fewer params, translation-tolerant pattern detectors; MLP on raw pixels explodes in size and ignores spatial structure.

**Q35. Role of pooling?** Downsample H,W (keep strongest signal with MaxPool), add small translation invariance, zero parameters, cut compute.

**Q36. Conv output formula + example?** O = floor((I-K+2P)/S)+1; 32,K5,P0,S1 -> 28.

**Q37. Params of Conv2d(3,6,5)?** (5*5*3+1)*6 = 456.

**Q38. Explain the class's CNN+RandomForest hybrid.** Trained CNN truncated at fc2 (84-d embedding); embeddings for all images feed a RandomForestClassifier; forest predicts classes; `feature_importances_` reveals which embedding dims matter.

**Q39. Why keep the batch dim when flattening?** `torch.flatten(x,1)` flattens from dim 1 so each sample stays a row; flattening dim 0 would merge the whole batch into one vector.

**Q40. joblib vs torch.save?** joblib pickles sklearn objects efficiently (numpy-heavy); torch.save serializes tensors/state_dicts. CNN -> torch.save; RandomForest -> joblib.dump.

[back to top](#table-of-contents)

## E. tracking — tensorboard and wandb (Q41-Q50)

**Q41. Why experiment tracking at all?** Reproducibility + comparison: many runs differ by hyperparams; logs let you see which change caused which metric move instead of guessing.

**Q42. Four SummaryWriter methods from class?** add_scalar(s), add_graph, add_pr_curve, add_hparams (SGPH). Always writer.close().

**Q43. How did class avoid runs overwriting each other?** create_writer built timestamped log dirs runs/date/experiment/model/extra — unique per run.

**Q44. Experiment design principle taught?** Change one variable at a time; start small (10% data, few epochs), scale what wins.

**Q45. wandb run lifecycle?** login -> init(project, config) -> log(dict per step) -> finish (ILAF).

**Q46. Three sweep-config keys?** method, metric{name,goal}, parameters (MMP).

**Q47. grid vs random vs bayes?** grid = exhaustive over discrete values; random = independent draws (great baseline, handles continuous); bayes = model-guided proposals using past results.

**Q48. What does wandb.agent do?** Pulls the next hyperparameter combo from the sweep server and calls your train function with it, `count` times; config values arrive via wandb.config.

**Q49. What is Hyperband early termination?** Bandit-style scheduler that stops clearly-underperforming runs at checkpoints (min_iter), reallocating budget to promising ones.

**Q50. Artifact producer vs consumer calls?** Producer: `art = wandb.Artifact(name, type)`, `add_file`/`new_file`, `run.log_artifact(art)`. Consumer: `run.use_artifact("name:latest")`, `.download()`. Versions auto-increment only when content hash changes; graph view shows the pipeline DAG.

[back to top](#table-of-contents)

## F. distributed and deployment (Q51-Q60)

**Q51. One line to use multiple GPUs?** `model = nn.DataParallel(model)` (then .to(device)).

**Q52. Four internal ops of DataParallel?** replicate model -> scatter inputs -> parallel_apply -> gather outputs on device 0 (RSPG).

**Q53. Batch 30 on 2 GPUs — per-GPU size and output device?** 15 each; gathered output (30,...) on cuda:0.

**Q54. Data parallel vs model parallel?** DP: full model copied, batch split — for speed. MP: layers split across GPUs, activations flow between — for models too big for one GPU.

**Q55. Weakness of naive model parallel and the fix?** Only one GPU busy at a time; fix = pipeline micro-batches so stages overlap.

**Q56. DataParallel vs DistributedDataParallel?** DP = single process, multi-thread, GIL-limited, one machine. DDP = one process per GPU, gradient all-reduce, scales to multi-node — production choice.

**Q57. trace vs script?** trace records ops for one example input (fast, but freezes data-dependent branches, silently); script compiles the source, preserving if/loops. Rule: control flow -> script.

**Q58. What's inside a saved TorchScript file and why care?** Serialized code + weights, loadable without Python — e.g. C++ libtorch `torch::jit::load` for low-latency serving.

**Q59. Why does torch.onnx.export need a dummy input?** Export runs a trace of the graph; dummy defines shapes/dtypes. dynamic_axes then relaxes chosen dims (e.g. batch).

**Q60. Full ONNX serve sequence?** export -> onnx.load + checker.check_model -> onnxruntime.InferenceSession -> sess.run(None, {"input": numpy_array}) — NumPy in/out (EC-RI).

[back to top](#table-of-contents)

---
**Drill mode:** cover the answer, read the question aloud, answer in one breath. Anything you fumble, jump to the linked file via the nav above and reread that one section only.

[Hub](DLOPS_EXAM_00_Hub.md) | [Theory](DLOPS_EXAM_01_Theory.md) | [Numericals](DLOPS_EXAM_02_Numericals.md) | [Practice](DLOPS_EXAM_03_Practice.md) | [Deep Dive](DLOPS_EXAM_04_Notebook_Deep_Dive.md)
