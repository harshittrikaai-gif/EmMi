"""
Data mixing schedule.

Defines the phased data-source weighting strategy used during training:
  Phase 1  (0–400 K steps)   — foundation (English-heavy)
  Phase 2  (400 K–800 K)     — multilingual emphasis
  Phase 3  (800 K–1 M)       — vision-language integration
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class MixPhase:
    """A single phase of the data-mixing schedule."""
    name: str
    start_step: int
    end_step: int
    weights: Dict[str, float]  # source_name → weight (should sum to ~1.0)


# Default three-phase schedule from the architecture guide
DEFAULT_PHASES: List[MixPhase] = [
    MixPhase(
        name="phase1_foundation",
        start_step=0,
        end_step=400_000,
        weights={
            "en_web":    0.35,
            "en_code":   0.10,
            "en_books":  0.10,
            "en_papers": 0.10,
            "indic_web": 0.20,
            "indic_books": 0.05,
            "global_web": 0.10,
        },
    ),
    MixPhase(
        name="phase2_multilingual",
        start_step=400_000,
        end_step=800_000,
        weights={
            "en_web":    0.25,
            "en_code":   0.10,
            "en_books":  0.08,
            "en_papers": 0.12,
            "indic_web": 0.25,
            "indic_books": 0.08,
            "global_web": 0.12,
        },
    ),
    MixPhase(
        name="phase3_vision_language",
        start_step=800_000,
        end_step=1_000_000,
        weights={
            "en_web":          0.20,
            "indic_web":       0.20,
            "global_web":      0.10,
            "vision_language": 0.30,
            "en_code":         0.10,
            "en_papers":       0.10,
        },
    ),
]


class DataMixSchedule:
    """Look up the data-source weights for a given training step."""

    def __init__(self, phases: List[MixPhase] | None = None):
        self.phases = phases or DEFAULT_PHASES
        # Sort by start step
        self.phases.sort(key=lambda p: p.start_step)

    def get_weights(self, step: int) -> Dict[str, float]:
        """Return source weights for the given training *step*."""
        for phase in reversed(self.phases):
            if step >= phase.start_step:
                return phase.weights
        return self.phases[0].weights

    def get_phase_name(self, step: int) -> str:
        for phase in reversed(self.phases):
            if step >= phase.start_step:
                return phase.name
        return self.phases[0].name
