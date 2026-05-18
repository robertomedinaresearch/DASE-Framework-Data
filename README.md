# DASE — Deliberative Adaptive Stopping Ensemble

A routing layer for LLM ensembles that knows when to trust an answer and proves why.

DASE runs a multi-agent ensemble through an adaptive deliberation protocol and produces two outputs: a **verified answer** when the ensemble reaches structural consensus (98% accurate on commits), or a **flag for human review** when it doesn't. Every commit ships with a replayable audit trail.

The stopping rule is derived from evidence-accumulation models in computational neuroscience ([Medina, 2019](https://run.unl.pt/handle/10362/90973)).

📄 **Paper:** [arXiv:2605.04236v2](https://arxiv.org/abs/2605.04236v2)
📊 **Data & Logs:** [DASE-Framework-Data](https://github.com/robertomedinaresearch/DASE-Framework-Data)

## Installation

```bash
pip install dase-core
```

Or from source:

```bash
git clone https://github.com/robertomedinaresearch/dase-core.git
cd dase-core
pip install -e .
```

## Quickstart

### 1. Implement a worker backend

DASE is backend-agnostic. Implement the `WorkerBackend` protocol for your LLM provider:

```python
from dase_core import deliberate, ArenaConfig

class MyWorker:
    async def query(self, prompt, step, worker_id, prev_answers=None, temperature=0.2):
        # Call your LLM here. Return the raw response text.
        # Return "[ERR_...]" on failure.
        ...
```

### 2. Run deliberation

```python
import asyncio

async def main():
    workers = [
        MyWorker(model="model-a"),
        MyWorker(model="model-a"),
        MyWorker(model="model-a"),
        MyWorker(model="model-b"),  # diversity improves routing signal
        MyWorker(model="model-b"),
    ]

    result = await deliberate(
        prompt="Find the number of subsets of {1,2,...,10} that contain "
               "exactly one pair of consecutive integers.",
        workers=workers,
        cfg=ArenaConfig(half_width=4, alpha_weight=0.2),
    )

    if result.is_consensus:
        print(f"✅ Ship it: {result.answer}")
    else:
        print(f"⚠️ Flag for review: {result.answer}")

asyncio.run(main())
```

### 3. Use the routing signal

```python
# Binary routing decision
result.is_consensus      # True if right-wall commit (high confidence)
result.should_escalate   # True if anything other than consensus

# Commit metadata
result.commit_type       # "right_wall", "left_wall", or "fallback"
result.steps             # How many deliberation rounds were used
result.final_x           # Final arena position

# Full audit trail
for step in result.trajectory:
    print(f"t={step.step} x={step.x:+d} g={step.g:.3f} [{step.direction}]")
```

## How it works

DASE models ensemble deliberation as a spatial stopping problem. An agent starts at the centre of an arena and accumulates evidence across rounds:

```
    -W ─────────── 0 ─────────── +W
   (no consensus)  (start)    (consensus)
```

At each round, all workers are queried. A Bayesian evidence score `g` is computed from the vote split. The agent moves one step toward the wall with higher expected value, or waits. Commitment fires only on wall contact.

**Right-wall commit** (consensus reached): return the plurality answer with high confidence.

**Left-wall commit** (fragmented evidence): apply a global-frequency fallback and flag for review.

The key insight is that *adaptive stopping, not injection bandwidth, drives accuracy*. On AIME-300, the full 6.0 pp advantage over static-budget debate is attributable to stopping alone.

## Configuration

```python
ArenaConfig(
    half_width=8,       # Arena spans [-W, +W]. Larger W = deeper deliberation.
    cx=0.01,            # Movement cost. Robust across parameter sweeps.
    L=15,               # Maximum deliberation runway (steps).
    alpha_weight=0.2,   # How much infrastructure noise affects the evidence score.
)
```

**Choosing `half_width` (W):**

| Setting | W | Use case |
|---------|---|----------|
| Fast routing | 2-4 | When you want quick commits with moderate depth |
| Balanced | 4 | General-purpose (preferred on GPQA) |
| Deep deliberation | 8 | Hard reasoning tasks (preferred on AIME) |

W interacts with model capability. The paper found W=2 insufficient at 70B but effective at 120B. Start with W=4 and adjust based on your right-wall coverage.

## Key results from the paper

| Benchmark | Right-wall accuracy | Left-wall accuracy | Routing gap |
|-----------|--------------------|--------------------|-------------|
| GPQA-Extended (70B, W=4) | 81.1% | 41.5% | 39.5 pp |
| GPQA-Extended (70B, W=8) | 82.3% | 36.5% | 45.8 pp |
| AIME 2010-2023 (120B, W=2) | 98.3% | 72.8% | 25.5 pp |

DASE's routing signal is **complementary** to single-call verbalized confidence: the two mechanisms disagree on 37% of routing assignments (McNemar p=1.000), meaning a combined system covers more problems than either alone.

## Ensemble composition

Heterogeneous ensembles outperform homogeneous ones. The paper uses:

**70B tier:** 3x Qwen3-Next-80B-A3B + 2x Llama-3.3-70B

**120B tier:** 3x GPT-OSS-120B + 2x Qwen3-Next-80B-A3B

The minority model acts as an adversarial check that prevents premature consensus on wrong answers.

## Injection protocol

From round 2 onward, workers receive the extracted consensus answers from the previous round (~15 tokens) and are asked to verify and correct. The prompt template:

```
PREVIOUS ATTEMPTS BY OTHER MODELS:
- Candidate 1: {answer_1}
- Candidate 2: {answer_2}

INSTRUCTION: Review the previous consensus. Search specifically for
calculation errors or sign flips in the minority view.
```

Dense injection (~600 chars of full reasoning) is not required. Sparse injection achieves equivalent or better routing signal at one-tenth the bandwidth.

## Citation

```bibtex
@article{medina2026dase,
  title={Adaptive Consensus in LLM Ensembles via Sequential Evidence
         Accumulation: Automatic Budget Identification and Calibrated
         Commit Signals},
  author={Medina, Roberto},
  journal={arXiv preprint arXiv:2605.04236v2},
  year={2026}
}
```


## License

Data, logs, and evaluation outputs: MIT License (See [LICENSE](LICENSE) for details.
Source code in /code: Apache 2.0 (see code/LICENSE)
