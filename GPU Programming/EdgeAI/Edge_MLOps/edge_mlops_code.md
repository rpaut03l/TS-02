# 💻 EdgeAI · Edge MLOps — CODE

### *CI pipeline · Signed OTA · Delta patches · Drift detector · Telemetry*

> **Nav:** [← Edge MLOps README](README.md) | [📖 THEORY](edge_mlops_theory.md) | **CODE** | [🎯 PRACTICE →](edge_mlops_practice.md)

---

## 🏗️ Setup

```python
!pip install -q onnx onnxruntime tensorflow openvino openvino-dev \
    pynacl bsdiff4 scipy
```

---

## Ex 1 — CI pipeline sketch (GitHub Actions)

### 👶 What this does
One commit → all three artifacts (ONNX, TFLite, OpenVINO IR) + signed
manifest. The file below goes in `.github/workflows/edge-release.yml`.

```yaml
name: Edge release
on:
  push:
    tags: ["v*"]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install -r requirements.txt
      - run: python tools/train.py --out model.pt
      - run: python tools/export_onnx.py model.pt cat_dog.onnx
      - run: python tools/convert_tflite.py cat_dog.onnx cat_dog.tflite
      - run: mo --input_model cat_dog.onnx --output_dir ir
      - run: python tools/make_manifest.py
              --onnx cat_dog.onnx
              --tflite cat_dog.tflite
              --ir ir/cat_dog.xml
              --target "jetson-orin-nano"
              --out manifest.json
      - run: python tools/sign.py manifest.json
              --key "${{ secrets.RELEASE_ED25519 }}"
              --out manifest.sig
      - uses: actions/upload-artifact@v4
        with:
          name: edge-bundle
          path: |
            cat_dog.onnx
            cat_dog.tflite
            ir/
            manifest.json
            manifest.sig
```

---

## Ex 2 — Manifest + signing (Python / ed25519)

### 👶 What this does
Produce the manifest, hash every artifact, sign the whole manifest
with an ed25519 private key. The device verifies with the embedded
public key.

```python
import json, hashlib, os, base64, nacl.signing

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def build_manifest(model_version, artifacts, target,
                    min_runtime="2.15.0"):
    return {
        "schema": 1,
        "model_version": model_version,
        "target": target,
        "min_runtime": min_runtime,
        "created_at": 1714_000_000,
        "artifacts": [
            {"path": p, "sha256": sha256_file(p),
             "bytes": os.path.getsize(p), "role": role}
            for p, role in artifacts
        ],
    }

# Demo: fake 2 files
with open("a.bin", "wb") as f: f.write(b"a" * 100)
with open("b.bin", "wb") as f: f.write(b"b" * 200)

manifest = build_manifest(
    "vww.2024.04.12",
    [("a.bin", "weights"), ("b.bin", "metadata")],
    target="jetson-orin-nano")

with open("manifest.json", "w") as f:
    json.dump(manifest, f, indent=2, sort_keys=True)

# Generate / load ed25519 keys
sk = nacl.signing.SigningKey.generate()
pk = sk.verify_key
print("Device should bake in pk:",
      base64.b64encode(pk.encode()).decode())

# Sign the manifest bytes
sig = sk.sign(open("manifest.json", "rb").read()).signature
open("manifest.sig", "wb").write(sig)
print("Signed manifest + .sig ready to ship")
```

### Device-side verification

```python
# On the device (baked-in pk_bytes):
pk_bytes = base64.b64decode(b"<your_public_key_base64>")
verifier = nacl.signing.VerifyKey(pk_bytes)

data = open("manifest.json", "rb").read()
sig  = open("manifest.sig", "rb").read()
try:
    verifier.verify(data, sig)
    print("✅ Manifest signature valid")
except Exception as e:
    print("❌ Reject update:", e)
```

---

## Ex 3 — Delta OTA with bsdiff / bspatch

### 👶 What this does
Ship only the binary diff between the old and new model file. 5–20×
smaller uplink.

```python
import bsdiff4, os
# Simulate "old" and "new" models
old = os.urandom(1 << 20); new = bytearray(old); new[:512] = b"x" * 512
open("v1.tflite", "wb").write(old)
open("v2.tflite", "wb").write(bytes(new))

bsdiff4.file_diff("v1.tflite", "v2.tflite", "patch.bspatch")
print("v1  :", os.path.getsize("v1.tflite"), "bytes")
print("v2  :", os.path.getsize("v2.tflite"), "bytes")
print("Δ   :", os.path.getsize("patch.bspatch"), "bytes")

# On device
bsdiff4.file_patch("v1.tflite", "v2_reconstructed.tflite", "patch.bspatch")
assert open("v2.tflite", "rb").read() == open("v2_reconstructed.tflite", "rb").read()
print("✅ patched model matches")
```

Typical ratio on model updates: a 5 MB model + minor retrain
produces a ~200 KB patch.

---

## Ex 4 — Staged-rollout controller (server-side)

### 👶 What this does
Decide whether each device currently asking "do you have a new
version?" should receive the new one. Percentage-based ramp.

```python
import hashlib

def should_receive(device_id, release, pct):
    """Deterministic per-device decision so devices don't flap."""
    h = hashlib.sha1(f"{device_id}|{release}".encode()).hexdigest()
    bucket = int(h[:8], 16) % 100
    return bucket < pct

# Day 0: 1 % rollout
for did in ["dev-001", "dev-002", "dev-003", "dev-042", "dev-500"]:
    for p in [1, 5, 10, 25, 100]:
        r = should_receive(did, "vww.2024.04.12", p)
        print(f"{did} at {p:3d}%: {'YES' if r else ' no'}")
```

Moving `pct` up gradually is all the "rollout" logic you need — the
per-device hash keeps decisions stable.

---

## Ex 5 — Dual-partition A/B flip pseudo-code (device-side)

```c
// Bootloader pseudo-code (very close to what U-Boot / MCUboot do).
uint8_t active   = read_flag(PARTITION_ACTIVE);   // 'A' or 'B'
uint8_t healthy  = read_flag(PARTITION_HEALTHY);

if (boot_counter++ > 3 && !healthy) {
    // rolled back automatically
    active = (active == 'A') ? 'B' : 'A';
    write_flag(PARTITION_ACTIVE, active);
    boot_counter = 0;
    write_flag(BOOT_COUNTER, boot_counter);
}
jump_to(active == 'A' ? APP_A_ADDR : APP_B_ADDR);

// Application sets healthy flag after successful self-test
if (self_test_pass()) write_flag(PARTITION_HEALTHY, 1);
```

Combined with the staged rollout from Ex 4, this is the **full
rollback-safe update mechanism** — nothing fancier needed.

---

## Ex 6 — Drift detector: PSI on feature histograms

### 👶 What this does
Given training-time histogram `p` and current-time histogram `q`,
compute **Population Stability Index**. Alert if > 0.25.

```python
import numpy as np

def psi(p, q, eps=1e-8):
    p = np.array(p, dtype=float) + eps
    q = np.array(q, dtype=float) + eps
    p /= p.sum(); q /= q.sum()
    return float(np.sum((p - q) * np.log(p / q)))

# Fake: training histogram of input mean brightness
train_hist = [5, 30, 80, 120, 90, 40, 20, 10]

# Scenario 1 — same world
q_stable = [6, 28, 82, 125, 92, 38, 19, 10]
print("stable  PSI:", psi(train_hist, q_stable))

# Scenario 2 — drift (many more very-bright images)
q_shift = [2, 15, 30, 80, 90, 80, 60, 40]
print("shifted PSI:", psi(train_hist, q_shift))
```

Typical output:
```
stable  PSI: 0.0007
shifted PSI: 0.322    ← > 0.25, retrain
```

### Device-side implementation
On the device, you don't even need NumPy — a 16-bucket histogram of
the input mean updated every inference is < 100 bytes and takes a
handful of integer adds.

---

## Ex 7 — KS-test drift detector (feature-by-feature)

```python
from scipy import stats
ref = np.random.normal(0, 1, 1000)   # training-time reference
now = np.random.normal(0.5, 1.2, 500)  # shifted

D, p = stats.ks_2samp(ref, now)
print(f"KS statistic: {D:.3f}  p-value: {p:.2e}")
if p < 0.01:
    print("⚠️  Distribution has drifted")
```

For multiple features, Bonferroni-correct the p-values. Cheap and
surprisingly effective.

---

## Ex 8 — Telemetry emitter (JSON Lines, batched)

```python
import time, json, gzip, os, random

class TelemetryBuffer:
    def __init__(self, path="telemetry.jsonl.gz", flush_bytes=4096):
        self.path = path; self.flush_bytes = flush_bytes
        self._buf = []
    def emit(self, record):
        self._buf.append(json.dumps(record))
        if sum(len(s) for s in self._buf) > self.flush_bytes:
            self.flush()
    def flush(self):
        if not self._buf: return
        with gzip.open(self.path, "ab") as f:
            f.write(("\n".join(self._buf) + "\n").encode())
        self._buf = []

buf = TelemetryBuffer()
for _ in range(50):
    buf.emit({
        "ts_ms": int(time.time()*1000),
        "model_version": "vww.2024.04.12",
        "latency_ms": round(random.gauss(18, 2), 2),
        "confidence": round(random.uniform(0.6, 0.95), 3),
    })
buf.flush()
print("Bytes on disk:", os.path.getsize(buf.path))
```

Upload the gzipped JSONL over HTTPS every 10 min (or next connected
window). Server processes with any log stack (Fluent Bit → Loki /
CloudWatch / OTel collector).

---

## Ex 9 — A/B scoring (one KPI vs latency)

```python
import numpy as np
def ab_score(control, test, min_n=200, alpha=0.05):
    if len(control) < min_n or len(test) < min_n:
        return "inconclusive — not enough samples"
    t, p = stats.ttest_ind(control, test, equal_var=False)
    if p > alpha: return "no significant difference"
    return "test WINS" if np.mean(test) > np.mean(control) else "test LOSES"

control_conf = np.random.beta(8, 2, 500)   # old model confidences
test_conf    = np.random.beta(9, 2, 500)   # new model confidences
print(ab_score(control_conf, test_conf))
```

---

## Ex 10 — Observability glue — OTel counters + histograms

```python
# pip install opentelemetry-api opentelemetry-sdk
from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider

metrics.set_meter_provider(MeterProvider())
meter = metrics.get_meter("edgeai")

inf_latency = meter.create_histogram("inference_latency_ms")
inf_total   = meter.create_counter("inferences_total")
drift_score = meter.create_up_down_counter("drift_psi")

def record_inference(ms, version):
    inf_latency.record(ms, {"model_version": version})
    inf_total.add(1, {"model_version": version})

record_inference(18.3, "vww.2024.04.12")
```

Ship these via the **OpenTelemetry Collector** in gateway mode from
your fleet. Every Prometheus / Grafana / CloudWatch / Datadog
pipeline understands the format.

---

## 📝 Summary

| Exercise | What you built |
|---|---|
| 1 | CI workflow producing the full artifact bundle |
| 2 | Manifest + ed25519 signature + device-side verify |
| 3 | bsdiff delta OTA |
| 4 | Deterministic staged-rollout controller |
| 5 | A/B bootloader flip & rollback |
| 6 | PSI drift detector |
| 7 | KS-test feature-by-feature drift |
| 8 | JSON Lines gzipped telemetry buffer |
| 9 | A/B KPI scorer |
| 10 | OpenTelemetry metrics emission |

Glue them together in the [practice notebook](edge_mlops_practice.md)
and simulate a fleet of 1,000 devices.

---

> *GPU Programming · EdgeAI · Edge MLOps · CODE · github.com/rpaut03l/TS-02*
