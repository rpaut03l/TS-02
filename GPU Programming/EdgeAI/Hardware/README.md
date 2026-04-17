# 🛠️ EdgeAI · Hardware (beyond GPUs)

### *The non-GPU chips — NPUs, MCUs, FPGAs — plus the Edge-vs-Cloud GPU table*

> **Nav:** [← EdgeAI](../README.md) | [← Fundamentals](../Fundamentals/README.md) | [← GPU Types](../GPU_Types/README.md) | **Hardware** | [CUDA for Edge →](../CUDA_for_Edge/README.md)

---

## 👶 30-second story

GPUs are not the only chip in an edge device. Open up any modern
smartphone or smart camera and you'll find a **zoo**:

- A **NPU** that does only AI, 100× more efficiently than the GPU.
- A **DSP** for audio and signal processing.
- A **tiny MCU** that never sleeps and sips microwatts.
- Sometimes an **FPGA** for ultra-low-latency custom work.

This folder is that zoo tour — **everything on the edge that is not a
GPU**, plus the **Edge vs Cloud GPU** comparison that closes the
"where does Edge AI sit" story.

---

## 📁 Files in this folder

| File | What it is |
|---|---|
| [edge_ai_hardware_theory.md](edge_ai_hardware_theory.md) | The non-GPU atlas — NPUs (Apple ANE, Qualcomm Hexagon, Hailo, Coral), MCUs for TinyML (Cortex-M + Ethos-U55, ESP32, STM32, Pico), FPGAs at the edge (Zynq, Agilex), DSPs (Hexagon audio), the full **Edge vs Cloud GPU comparison table**, power & thermal budgets, benchmarks (MLPerf Tiny/Edge, TOPS/Watt leaderboard) |
| [edge_ai_hardware_code.md](edge_ai_hardware_code.md) | Runnable code — enumerate available accelerators (ONNX Runtime providers, OpenVINO devices), benchmark the same model across CPU / GPU delegate / simulated NPU, estimate energy per inference, parse MLPerf result CSVs |
| [edge_ai_hardware_practice.md](edge_ai_hardware_practice.md) | **Colab notebook** — build a "hardware fit chart" that plots TOPS/Watt vs price for every edge chip in a shared database, and matches chips to workloads under thermal + bandwidth caps |

---

## 🎯 After reading this you should be able to…

- Name **4 kinds of chip** (GPU, NPU, MCU, FPGA) and say what each is
  best at
- List the **top 5 Edge NPUs** and the runtime each one uses
- Draw the **Edge vs Cloud GPU comparison table** from memory
  (Primary Goal · Location · Typical Power · Latency · Cost · Memory)
- Pick the right **MCU** for a TinyML product under 1 W
- Read a **MLPerf Tiny / Edge** submission and understand what it
  measured
- Reason about **thermal throttling** for a fanless edge box

---

> *GPU Programming · EdgeAI · Hardware · github.com/rpaut03l/TS-02*
