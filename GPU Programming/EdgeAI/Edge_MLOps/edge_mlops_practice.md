# 🎯 EdgeAI · Edge MLOps — PRACTICE

### *Simulate a 1,000-device fleet · push an OTA · watch drift · auto-rollback*

> **Nav:** [← Edge MLOps README](README.md) | [📖 THEORY](edge_mlops_theory.md) | [💻 CODE](edge_mlops_code.md) | **PRACTICE**

---

## 🎯 What you'll build

A Colab notebook that simulates a **fleet of 1,000 devices** and
exercises a full release cycle:

1. Build + sign a model bundle.
2. Roll out to 10 % of the fleet (canary).
3. Devices report latency + drift telemetry.
4. A monitor checks green/red criteria.
5. Promote to 100 % or **roll back**.
6. Produce a human-readable release report.

---

## Cell 1 — Setup

```python
!pip install -q pynacl scipy matplotlib
import random, time, hashlib, json, base64, statistics
import numpy as np, matplotlib.pyplot as plt
from scipy import stats
import nacl.signing
random.seed(7); np.random.seed(7)
```

---

## Cell 2 — Device simulator

```python
class FakeDevice:
    def __init__(self, did, baseline_latency):
        self.did = did
        self.model = "v1.0.0"
        self.baseline_latency = baseline_latency   # chip-specific baseline
        self.drift_state = 0.0                     # will shift over time
    def infer(self):
        # latency is chip-baseline + Gaussian noise + model overhead
        overhead = 3.0 if self.model == "v2.0.0" else 0.0
        ms = self.baseline_latency + overhead + random.gauss(0, 1)
        # a "feature mean" that could drift over time
        feat = random.gauss(0.5 + self.drift_state, 0.1)
        return dict(ms=ms, feat=feat, model=self.model)

fleet = [FakeDevice(f"dev-{i:04d}", random.uniform(10, 15))
         for i in range(1000)]
print("Fleet size:", len(fleet))
```

---

## Cell 3 — Build + sign a release manifest

```python
def sha256_bytes(b): return hashlib.sha256(b).hexdigest()

model_bytes_v2 = b"fake model v2 weights..."
manifest = {
    "schema": 1,
    "model_version": "v2.0.0",
    "target": "demo-edge",
    "min_runtime": "2.15.0",
    "created_at": int(time.time()),
    "artifacts": [{"path": "model.tflite",
                   "sha256": sha256_bytes(model_bytes_v2),
                   "bytes": len(model_bytes_v2),
                   "role": "weights"}]
}
manifest_bytes = json.dumps(manifest, sort_keys=True).encode()

sk = nacl.signing.SigningKey.generate()
pk = sk.verify_key
sig = sk.sign(manifest_bytes).signature
print("Signed. Public key (base64):",
      base64.b64encode(pk.encode()).decode()[:20], "…")
```

---

## Cell 4 — Staged rollout + device-side verify

```python
def should_receive(did, release, pct):
    h = hashlib.sha1(f"{did}|{release}".encode()).hexdigest()
    return int(h[:8], 16) % 100 < pct

def device_apply(device, manifest_bytes, sig, model_bytes):
    try:
        pk.verify(manifest_bytes, sig)            # signature check
    except Exception:
        return False
    m = json.loads(manifest_bytes)
    if m["artifacts"][0]["sha256"] != sha256_bytes(model_bytes):
        return False
    device.model = m["model_version"]
    return True

# 10 % canary
canary = [d for d in fleet if should_receive(d.did, "v2.0.0", 10)]
print("Canary size:", len(canary))
ok = sum(device_apply(d, manifest_bytes, sig, model_bytes_v2)
         for d in canary)
print(f"Applied update on {ok}/{len(canary)} canary devices")
```

---

## Cell 5 — Simulate one hour of traffic and emit telemetry

```python
def hour_of_inferences(n_infer=200):
    rows = []
    for d in fleet:
        for _ in range(n_infer):
            r = d.infer()
            rows.append(dict(did=d.did, **r))
    return rows

rows = hour_of_inferences(100)
print("Telemetry rows:", len(rows))

# Split by model version
v1 = [r for r in rows if r["model"] == "v1.0.0"]
v2 = [r for r in rows if r["model"] == "v2.0.0"]
print(f"v1 samples: {len(v1)}  v2 samples: {len(v2)}")
```

---

## Cell 6 — Green / red gate

```python
def p99(x): return np.percentile(x, 99)

lat_v1 = [r["ms"] for r in v1]
lat_v2 = [r["ms"] for r in v2]

gate = dict(
    p99_ratio   = p99(lat_v2) / p99(lat_v1),
    mean_ratio  = np.mean(lat_v2) / np.mean(lat_v1),
    ks_feat     = stats.ks_2samp([r["feat"] for r in v1],
                                  [r["feat"] for r in v2]).statistic,
)
print("Gate metrics:", gate)
green = (gate["p99_ratio"] < 1.3 and gate["mean_ratio"] < 1.3
         and gate["ks_feat"] < 0.15)
print("🟢 Promote" if green else "🔴 Rollback")
```

---

## Cell 7 — If green: ramp to 100 %; if red: auto-rollback

```python
if green:
    # Ramp
    for pct in [25, 50, 100]:
        for d in fleet:
            if d.model != "v2.0.0" and should_receive(d.did, "v2.0.0", pct):
                device_apply(d, manifest_bytes, sig, model_bytes_v2)
        print(f"Ramped to {pct}% — on v2.0.0:",
              sum(1 for d in fleet if d.model == "v2.0.0"))
else:
    # Rollback
    for d in fleet:
        d.model = "v1.0.0"
    print("Rolled back all devices to v1.0.0")
```

---

## Cell 8 — Inject drift and re-evaluate

```python
# Simulate concept drift: sensor slowly biases over the next few weeks
for d in fleet:
    d.drift_state += random.uniform(0.0, 0.3)

rows_after = hour_of_inferences(100)
ref = [r["feat"] for r in rows]          # pre-drift
cur = [r["feat"] for r in rows_after]    # post-drift
psi_stat = stats.ks_2samp(ref, cur)
print(f"KS statistic: {psi_stat.statistic:.3f}  "
      f"p-value: {psi_stat.pvalue:.2e}")
if psi_stat.statistic > 0.2:
    print("⚠️  Drift detected — trigger retraining pipeline")
```

---

## Cell 9 — Human-readable release report

```python
def release_report(canary_rows, full_rows, gate, decision):
    v1l = [r["ms"] for r in canary_rows if r["model"] == "v1.0.0"]
    v2l = [r["ms"] for r in canary_rows if r["model"] == "v2.0.0"]
    return f"""# Release Report — v2.0.0

**Canary size:** {sum(1 for r in canary_rows if r['model']=='v2.0.0'):,} devices
**Decision:**    {decision}

## Latency
- v1 mean / p99: {np.mean(v1l):.2f} / {p99(v1l):.2f} ms
- v2 mean / p99: {np.mean(v2l):.2f} / {p99(v2l):.2f} ms
- p99 ratio:    {gate['p99_ratio']:.3f}  (threshold 1.3)

## Drift
- KS feature statistic (v1 vs v2): {gate['ks_feat']:.3f} (threshold 0.15)

## Fleet
- On v2.0.0:  {sum(1 for d in fleet if d.model == 'v2.0.0'):,} / {len(fleet):,}
"""

print(release_report(rows, rows_after, gate,
                     "PROMOTED" if green else "ROLLED BACK"))
```

---

## Cell 10 — Stretch goals

1. Add a **crash-rate** gate that fails the rollout if > 0.1 % of
   canary devices raise an error.
2. Replace the global latency gate with **per-device** gates — a
   single slow device shouldn't fail the whole rollout.
3. Add **telemetry batching** (JSONL + gzip) and upload only when
   the simulated network is "up" (probability 0.7 per device).
4. Simulate a **50 % WAN outage** during the rollout — does your
   controller still make sane decisions?
5. Plug the drift alert into a synthetic **retraining job** that
   produces `v2.0.1` and loops back to Cell 3.

---

## 🎓 What you should take away

- A full Edge MLOps cycle fits in **one notebook** — it's the
  *discipline* that's hard, not the code.
- **Signatures + hashes + staged rollouts + gates + rollback** are a
  fixed recipe. Learn it once, write the scaffold, reuse forever.
- **Drift monitoring is free** if you emit tiny histograms. Skipping
  it is never justified.
- The **release report** is the artefact — it's what you show your
  team, not the metrics dashboard.

Next: [**Security & Privacy →**](../Security_Privacy/README.md) — the
final layer of the edge AI stack.

---

> *GPU Programming · EdgeAI · Edge MLOps · PRACTICE · github.com/rpaut03l/TS-02*
