"""
Example: Using DASE with OpenAI-compatible endpoints.

This demonstrates how to implement the WorkerBackend protocol
for any LLM provider. Replace the backend with your own.
"""

import asyncio
from dase_core import deliberate, ArenaConfig


# ── Implement the WorkerBackend protocol for your provider ──────────

class OpenAIWorker:
    """Example backend for OpenAI-compatible APIs."""

    def __init__(self, model: str, api_key: str, base_url: str | None = None):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url or "https://api.openai.com/v1"

    async def query(
        self,
        prompt: str,
        step: int,
        worker_id: int,
        prev_answers: list[str] | None = None,
        temperature: float = 0.2,
    ) -> str:
        import httpx

        content = f"TASK: {prompt}\n\n"

        if step > 1 and prev_answers:
            content += "PREVIOUS ATTEMPTS BY OTHER MODELS:\n"
            for i, ans in enumerate(prev_answers):
                content += f"- Candidate {i+1}: {ans}\n"
            content += (
                "\nINSTRUCTION: Review the previous consensus. Search "
                "specifically for calculation errors or sign flips in "
                "the minority view. If the current consensus is an "
                "integer, re-calculate the final step using an "
                "alternative method to verify.\n"
            )

        content += "\nCOMMAND: Conclude with: ### FINAL_CONSENSUS: [Answer]"

        async with httpx.AsyncClient(timeout=180.0) as client:
            try:
                resp = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={
                        "model": self.model,
                        "messages": [{"role": "user", "content": content}],
                        "max_tokens": 4096,
                        "temperature": temperature,
                    },
                )
                if resp.status_code == 200:
                    return resp.json()["choices"][0]["message"]["content"]
                return f"[ERR_HTTP_{resp.status_code}]"
            except Exception as e:
                return f"[ERR_{type(e).__name__}]"


# ── Run deliberation ────────────────────────────────────────────────

async def main():
    # Configure a mixed ensemble (diversity improves routing signal)
    workers = [
        OpenAIWorker("gpt-4o", api_key="YOUR_KEY"),
        OpenAIWorker("gpt-4o", api_key="YOUR_KEY"),
        OpenAIWorker("gpt-4o", api_key="YOUR_KEY"),
        OpenAIWorker("gpt-4o-mini", api_key="YOUR_KEY"),
        OpenAIWorker("gpt-4o-mini", api_key="YOUR_KEY"),
    ]

    # Arena configuration (see paper for benchmark-specific recommendations)
    cfg = ArenaConfig(
        half_width=4,       # W=4 for GPQA-style tasks; W=8 for harder math
        cx=0.01,            # movement cost (robust across parameter sweeps)
        L=15,               # deliberation runway
        alpha_weight=0.2,   # noise-floor trust
    )

    result = await deliberate(
        prompt="Find the number of subsets of {1,2,...,10} that contain "
               "exactly one pair of consecutive integers.",
        workers=workers,
        cfg=cfg,
    )

    # ── Use the routing signal ──────────────────────────────────────

    print(f"Answer:      {result.answer}")
    print(f"Commit type: {result.commit_type}")
    print(f"Steps:       {result.steps}")
    print(f"Final x:     {result.final_x}")
    print()

    if result.is_consensus:
        print("✅ CONSENSUS — route to production")
    else:
        print("⚠️  NO CONSENSUS — flag for human review")

    # ── Audit trail ─────────────────────────────────────────────────

    print("\nDeliberation trajectory:")
    for step in result.trajectory:
        print(f"  t={step.step:2d}  x={step.x:+3d}  g={step.g:.3f}  "
              f"V_R={step.v_right:+.3f}  V_L={step.v_left:+.3f}  "
              f"[{step.direction}]  "
              f"candidate='{step.candidate_right}'")


if __name__ == "__main__":
    asyncio.run(main())
