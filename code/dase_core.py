"""
DASE — Deliberative Adaptive Stopping Ensemble (Core Algorithm)
================================================================

Reference implementation of the DASE-Spatial stopping rule for LLM
ensemble deliberation, as described in:

    Medina, R. (2026). "Adaptive Consensus in LLM Ensembles via Sequential
    Evidence Accumulation: Automatic Budget Identification and Calibrated
    Commit Signals." arXiv:2605.04236v2.

The stopping rule is derived from the embodied optimal-stopping model in:

    Medina, R. E. (2019). "Optimal Decision Making for Temporally Extended
    Actions." PhD thesis, ITQB NOVA / Champalimaud Centre for the Unknown.

This module contains the algorithm only — no API clients, no billing,
no infrastructure. Plug in any LLM backend by implementing the
`WorkerBackend` protocol.

License: Apache 2.0
"""

from __future__ import annotations

import re
import functools
from collections import Counter
from dataclasses import dataclass, field
from typing import Protocol, Sequence

import numpy as np
from scipy.special import betainc


# ═══════════════════════════════════════════════════════════════════════
# 1. HEURISTIC EVIDENCE SCORE  (Eq. 3.11, Medina 2019)
# ═══════════════════════════════════════════════════════════════════════

def compute_g(
    r: int,
    n: int,
    alpha: float,
    alpha_weight: float = 0.2,
) -> float:
    """
    Bayesian posterior belief that the plurality answer is genuinely
    dominant, given the observed vote split and an observable noise floor.

    This is Eq. 3.11 from Medina (2019), with a tunable noise-floor
    trust parameter.

    Parameters
    ----------
    r : int
        Votes for the plurality answer this step.
    n : int
        Total workers called this step (including empty/error returns).
    alpha : float
        empty_count / n — observable noise floor in [0, 1).
    alpha_weight : float
        Trust parameter in [0, 1] controlling how much of the
        infrastructure noise floor is treated as true signal noise.
            1.0   → full Bayesian correction (original Eq. 3.11)
            0.0   → simple vote fraction over valid responses
            (0,1) → interpolation between the two

    Returns
    -------
    float
        g in [0, 1].  g > 0.5 favours consensus (right wall);
        g < 0.5 favours no-consensus (left wall); g ≈ 0.5 is uncertain.
        When all workers return empty, g = 0.0 (full uncertainty toward
        no-consensus).

    Notes
    -----
    The evidence score uses the regularised incomplete beta function:

        g = [I_{1-wα}(r+1, n-r+1) - I_{0.5}(r+1, n-r+1)]
          / [I_{1-wα}(r+1, n-r+1) - I_{wα}(r+1, n-r+1)]

    where w = alpha_weight and α = alpha.
    """
    alpha_eff = float(np.clip(alpha_weight * alpha, 1e-6, 0.5 - 1e-6))

    if n <= 0:
        return 0.5
    r = int(np.clip(r, 0, n))

    a, b = r + 1, n - r + 1

    I_hi = betainc(a, b, 1.0 - alpha_eff)   # upper tail
    I_mid = betainc(a, b, 0.5)               # neutral midpoint
    I_lo = betainc(a, b, alpha_eff)          # lower tail

    den = I_hi - I_lo
    if den < 1e-12:
        return 0.5
    return float(np.clip((I_hi - I_mid) / den, 0.0, 1.0))


# ═══════════════════════════════════════════════════════════════════════
# 2. SPATIAL ARENA CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ArenaConfig:
    """
    Parameters for the DASE-Spatial stopping rule.

    Attributes
    ----------
    half_width : int
        Arena spans [-half_width, +half_width]. Controls deliberation
        depth. W=4 preferred on GPQA; W=8 on AIME (see paper §5).
    cx : float
        Movement cost per unit displacement (thesis value: 0.01).
    L : int
        Maximum deliberation runway (steps before forced fallback).
    alpha_weight : float
        Noise-floor trust parameter for compute_g(). Default 0.2.
    """
    half_width: int = 8
    cx: float = 0.01
    L: int = 15
    alpha_weight: float = 0.2


# ═══════════════════════════════════════════════════════════════════════
# 3. TERMINAL-COMMIT VALUES AND DECISION POLICY  (Eq. 3.3, Medina 2019)
# ═══════════════════════════════════════════════════════════════════════

def v_right(g: float, x: int, cfg: ArenaConfig) -> float:
    """
    Terminal value for committing to the consensus (right) wall.
    V_R = g - cx * (W - x)
    """
    return g - cfg.cx * (cfg.half_width - x)


def v_left(g: float, x: int, cfg: ArenaConfig) -> float:
    """
    Terminal value for committing to the no-consensus (left) wall.
    V_L = (1 - g) - cx * (x + W)
    """
    return (1.0 - g) - cfg.cx * (x + cfg.half_width)


def spatial_step(g: float, x: int, cfg: ArenaConfig) -> tuple[int, str]:
    """
    Single spatial decision step.

    Move one step toward the higher-value wall if that value > 0,
    or wait if both terminal values are non-positive. Commitment
    fires only on wall contact — it emerges from trajectory, not
    instant thresholding.

    Parameters
    ----------
    g : float
        Current evidence score from compute_g().
    x : int
        Current arena position.
    cfg : ArenaConfig
        Arena parameters.

    Returns
    -------
    (new_x, direction) where direction is one of:
        "move_right", "move_left", "wait"
    """
    vr = v_right(g, x, cfg)
    vl = v_left(g, x, cfg)

    if vr > 0 or vl > 0:
        if vr >= vl:
            return x + 1, "move_right"
        else:
            return x - 1, "move_left"
    return x, "wait"


# ═══════════════════════════════════════════════════════════════════════
# 4. ANSWER NORMALISATION
# ═══════════════════════════════════════════════════════════════════════

_RE_TEXT = re.compile(r'\\?text\{[^\}]+\}')
_RE_START_VAR = re.compile(r'^[a-zA-Z]\s*=\s*')
_RE_COORDS = re.compile(r'[\(\[\{]\s*-?[\d\.]+\s*,\s*-?[\d\.]+\s*[\)\]\}]')
_RE_JUNK = re.compile(r'[\$\{\}\[\]\\]')
_RE_MULTI_COMMA = re.compile(r',\s+')


def _expand_frac(s: str) -> str:
    """Expand LaTeX \\frac{a}{b} to (a)/(b)."""
    while True:
        m = re.search(r'\\?frac\s*\{', s)
        if not m:
            break
        start = m.start()
        brace_start = m.end() - 1

        depth, num_end = 0, None
        for i in range(brace_start, len(s)):
            if s[i] == '{':
                depth += 1
            elif s[i] == '}':
                depth -= 1
                if depth == 0:
                    num_end = i
                    break
        if num_end is None:
            break
        numerator = s[brace_start + 1: num_end]

        den_start = den_end = None
        denominator = None
        for j in range(num_end + 1, len(s)):
            if s[j] == '{':
                den_start = j
                break
            elif not s[j].isspace():
                den_start = den_end = j
                denominator = s[j]
                break
        else:
            break

        if s[den_start] == '{':
            depth, den_end = 0, None
            for i in range(den_start, len(s)):
                if s[i] == '{':
                    depth += 1
                elif s[i] == '}':
                    depth -= 1
                    if depth == 0:
                        den_end = i
                        break
            if den_end is None:
                break
            denominator = s[den_start + 1: den_end]

        s = s[:start] + f'({numerator})/({denominator})' + s[den_end + 1:]
    return s


def _expand_sqrt(s: str) -> str:
    """Expand LaTeX \\sqrt{x} to (x)**0.5."""
    while True:
        m = re.search(r'\\?sqrt\s*(?=[\{\(])', s)
        if not m:
            break
        start = m.start()
        open_pos = m.end()
        open_ch = s[open_pos]
        close_ch = '}' if open_ch == '{' else ')'

        depth, inner_end = 0, None
        for i in range(open_pos, len(s)):
            if s[i] == open_ch:
                depth += 1
            elif s[i] == close_ch:
                depth -= 1
                if depth == 0:
                    inner_end = i
                    break
        if inner_end is None:
            break

        inner = s[open_pos + 1: inner_end]
        s = s[:start] + f'({inner})**0.5' + s[inner_end + 1:]
    return s


@functools.lru_cache(maxsize=4096)
def normalize(s: str, compact: bool = True) -> str:
    """
    Normalize an LLM-extracted answer for comparison.

    Handles LaTeX fractions/sqrt, MC letter answers, comma-separated
    thousands, coordinate pairs, and common symbolic representations.
    """
    if not s or str(s).lower().strip() == "empty":
        return "empty"
    s = str(s).lower().strip()
    s = _RE_TEXT.sub('', s)
    s = _RE_START_VAR.sub('', s)
    s = _expand_frac(s)
    s = _expand_sqrt(s)
    s = s.replace(r'\pm', '+/-').replace('±', '+/-')

    if re.match(r'^\(?[a-g]\)?$', s):
        return s.replace('(', '').replace(')', '').strip()

    s = re.sub(r'(?<=\d),(?=\d{3}(?!\d))', '', s)

    if _RE_COORDS.search(s):
        s = re.sub(r'[\(\)\[\]\{\}]', '', s)
        s = f'({s})'

    s = _RE_MULTI_COMMA.sub(',', s)
    if not re.search(r'\d,\d', s):
        s = s.replace(',', '')
    s = _RE_JUNK.sub('', s)
    for term in ['left', 'right', 'boxed', 'degrees', 'circ']:
        s = re.sub(rf'\\?{term}', '', s)

    s = s.replace('π', 'pi').replace('\\pi', 'pi')
    s = s.replace('τ', 'tau').replace('\\tau', 'tau')
    s = s.replace('∞', 'oo').replace('\\infty', 'oo')

    return s.replace(' ', '') if compact else s


def extract_answer(text: str) -> str:
    """
    Extract a final answer from an LLM response.

    Looks for ### FINAL_CONSENSUS: [answer] or ### FINAL_ANSWER: [answer],
    falling back to the last bracketed expression.
    """
    if not text or "[ERR_" in text:
        return "empty"
    for pattern in [
        r"###\s*FINAL_CONSENSUS\s*:\s*\[?([^\]\n]+)\]?",
        r"###\s*FINAL_ANSWER\s*:\s*\[?([^\]\n]+)\]?",
    ]:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return m.group(1).strip().strip('[]')
    fallback = re.findall(r"\[([^\]]{1,50})\]", text)
    return fallback[-1].strip() if fallback else "empty"


# ═══════════════════════════════════════════════════════════════════════
# 5. DELIBERATION ENGINE (backend-agnostic)
# ═══════════════════════════════════════════════════════════════════════

class WorkerBackend(Protocol):
    """
    Protocol for LLM worker backends. Implement this to plug in any
    LLM provider (OpenAI, Anthropic, Databricks, local models, etc.).
    """
    async def query(
        self,
        prompt: str,
        step: int,
        worker_id: int,
        prev_answers: list[str] | None = None,
        temperature: float = 0.2,
    ) -> str:
        """
        Send a prompt to a worker and return the raw response text.
        Return "[ERR_...]" on failure.
        """
        ...


@dataclass
class StepRecord:
    """Record for a single deliberation step."""
    step: int
    x: int
    g: float
    alpha: float
    v_right: float
    v_left: float
    direction: str
    candidate_right: str
    candidate_left: str
    raw_answers: list[str]


@dataclass
class DASEResult:
    """Result of a DASE deliberation."""
    answer: str
    commit_type: str          # "right_wall", "left_wall", or "fallback"
    steps: int
    final_x: int
    trajectory: list[StepRecord] = field(default_factory=list)
    global_counts: dict[str, int] = field(default_factory=dict)

    @property
    def is_consensus(self) -> bool:
        """True if the ensemble reached structural consensus."""
        return self.commit_type == "right_wall"

    @property
    def should_escalate(self) -> bool:
        """True if this result should be flagged for human review."""
        return not self.is_consensus


async def deliberate(
    prompt: str,
    workers: Sequence[WorkerBackend],
    cfg: ArenaConfig = ArenaConfig(),
) -> DASEResult:
    """
    Run the DASE-Spatial deliberation protocol.

    This is the core algorithm. It queries the worker ensemble across
    multiple rounds, accumulates evidence via the spatial arena, and
    commits when a wall is reached or the runway is exhausted.

    Parameters
    ----------
    prompt : str
        The question to deliberate on.
    workers : Sequence[WorkerBackend]
        LLM worker backends (typically 3-5 diverse models).
    cfg : ArenaConfig
        Arena configuration parameters.

    Returns
    -------
    DASEResult
        Contains the answer, commit type, trajectory, and all
        metadata needed for the audit trail.

    Examples
    --------
    >>> from dase_core import deliberate, ArenaConfig
    >>> # Implement your own WorkerBackend for your LLM provider
    >>> result = await deliberate("What is 2+2?", workers)
    >>> if result.is_consensus:
    ...     print(f"Committed: {result.answer}")
    ... else:
    ...     print(f"Flagged for review: {result.answer}")
    """
    import asyncio

    n_workers = len(workers)
    current_x = 0
    global_counter: Counter = Counter()
    trajectory: list[StepRecord] = []

    answer = "empty"
    commit_type = "fallback"

    for t in range(1, cfg.L + 1):
        # Determine what to inject from previous round
        prev_answers = None
        if trajectory:
            prev_norms = trajectory[-1].raw_answers
            prev_answers = [a for a in prev_norms
                           if a != "empty" and "[ERR_" not in a]

        # Query all workers in parallel
        temperature = 0.2 if t == 1 else 0.4
        tasks = [
            w.query(prompt, t, i, prev_answers, temperature)
            for i, w in enumerate(workers)
        ]
        raw_responses = await asyncio.gather(*tasks)

        # Extract and normalise answers
        norms = [normalize(extract_answer(r)) for r in raw_responses]
        valid = [n for n in norms if n != "empty" and "[ERR_" not in n]
        global_counter.update(valid)

        # Compute evidence score
        empty_count = sum(
            1 for n in norms if n == "empty" or "[ERR_" in n
        )
        alpha = empty_count / n_workers if n_workers > 0 else 0.5

        if not valid:
            g = 0.0
            candidate_right = "empty"
            candidate_left = "empty"
        else:
            counts = Counter(valid)
            top2 = counts.most_common(2)
            candidate_right = top2[0][0]
            candidate_left = top2[1][0] if len(top2) > 1 else "empty"
            r_votes = top2[0][1]
            g = compute_g(r_votes, n_workers, alpha, cfg.alpha_weight)

        # Spatial step
        vr = v_right(g, current_x, cfg)
        vl = v_left(g, current_x, cfg)
        current_x, direction = spatial_step(g, current_x, cfg)

        trajectory.append(StepRecord(
            step=t, x=current_x, g=round(g, 4), alpha=round(alpha, 4),
            v_right=round(vr, 4), v_left=round(vl, 4),
            direction=direction,
            candidate_right=candidate_right,
            candidate_left=candidate_left,
            raw_answers=norms,
        ))

        # Terminal check: wall contact
        if current_x >= cfg.half_width:
            answer = candidate_right
            commit_type = "right_wall"
            break

        if current_x <= -cfg.half_width:
            answer = (global_counter.most_common(1)[0][0]
                      if global_counter else "no_consensus")
            commit_type = "left_wall"
            break

    # Runway exhausted
    if commit_type == "fallback":
        answer = (global_counter.most_common(1)[0][0]
                  if global_counter else "empty")

    return DASEResult(
        answer=answer,
        commit_type=commit_type,
        steps=len(trajectory),
        final_x=current_x,
        trajectory=trajectory,
        global_counts=dict(global_counter),
    )


# ═══════════════════════════════════════════════════════════════════════
# 6. CONVENIENCE: MINIMUM BELIEF THRESHOLD
# ═══════════════════════════════════════════════════════════════════════

def min_belief_for_commit(x: int, cfg: ArenaConfig) -> float:
    """
    Minimum evidence score g*(x) required for the agent to move
    rightward (toward consensus) from position x.

    g*(x) = cx * (W - x)

    This defines the effective collapsing threshold: as x approaches
    the right wall, less evidence is needed to continue rightward.
    """
    return cfg.cx * (cfg.half_width - x)
