"""
Emotion / Frustration State Manager for the Customer Simulator.
Scale: 1 to 5
1 = Very calm / Satisfied
2 = Slightly concerned
3 = Moderately frustrated
4 = Highly frustrated
5 = Extremely angry / About to escalate
"""

from typing import Dict, Optional
from dataclasses import dataclass, field
import re

from config import EMOTION_SCALE_MIN, EMOTION_SCALE_MAX


EMOTION_LABELS = {
    1: "Very calm / Satisfied",
    2: "Slightly concerned",
    3: "Moderately frustrated",
    4: "Highly frustrated",
    5: "Extremely angry / About to escalate",
}


POSITIVE_SIGNALS = [
    r"\b(refund|full refund|process(ed|ing)? (the )?refund)\b",
    r"\b(apologi[sz]e|sorry for the (inconvenience|delay|trouble))\b",
    r"\b(escalat(e|ing|ed)|manager|supervisor)\b",
    r"\b(confirm(ed|ing)?|right away|immediately|within \d+ (hours?|days?))\b",
    r"\b(compensation|discount|credit|voucher)\b",
    r"\b(understood|I see the issue|you're right)\b",
    r"\b(I (will|can) (help|fix|take care|resolve))\b",
]

NEGATIVE_SIGNALS = [
    r"\b(policy|unfortunately we cannot|not possible|unable to)\b",
    r"\b(please wait|another \d+ days?|try again later)\b",
    r"\b(contact (the )?carrier|outside our control)\b",
    r"\b(no refund|store credit only|final sale)\b",
    r"\b(you (need|must|have) to)\b",
    r"\b(I don't have (access|permission|authority))\b",
    r"\b(not my problem|not our fault|deal with it)\b",
]


@dataclass
class EmotionState:
    intensity: int = 3
    label: str = "Moderately frustrated"
    history: list = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "intensity": self.intensity,
            "label": self.label,
            "history": self.history[-8:],
        }


class EmotionManager:

    def __init__(
        self,
        initial_emotion: str = "frustrated",
        initial_intensity: Optional[int] = None,
        persona_modifier: int = 0,
    ):
        self.persona_modifier = persona_modifier

        emotion_start_map = {
            "calm": 1,
            "polite": 1,
            "satisfied": 1,
            "mildly_concerned": 2,
            "concerned": 2,
            "confused": 2,
            "frustrated": 3,
            "moderately_frustrated": 3,
            "impatient": 4,
            "angry": 4,
            "highly_frustrated": 4,
            "furious": 5,
            "extremely_angry": 5,
        }

        if initial_intensity is not None:
            start = initial_intensity
        else:
            start = emotion_start_map.get(
                initial_emotion.lower(),
                3
            )

        start = max(
            EMOTION_SCALE_MIN,
            min(EMOTION_SCALE_MAX, start)
        )

        self.state = EmotionState(
            intensity=start,
            label=EMOTION_LABELS[start],
            history=[
                f"init → {start}/5 ({EMOTION_LABELS[start]})"
            ],
        )

    def get_state(self) -> EmotionState:
        return self.state

    def _score_agent_message(self, agent_message: str) -> int:
        if not agent_message or not agent_message.strip():
            return -2

        text = agent_message.lower()
        score = 0

        for pat in POSITIVE_SIGNALS:
            if re.search(pat, text, re.IGNORECASE):
                score += 1

        for pat in NEGATIVE_SIGNALS:
            if re.search(pat, text, re.IGNORECASE):
                score -= 1

        if len(agent_message.split()) < 5:
            score -= 0.5

        if any(
            w in text
            for w in [
                "i understand",
                "i can see",
                "that must be",
                "frustrating",
                "sorry",
                "apologize",
            ]
        ):
            score += 0.5

        return int(round(score))

    def update(self, agent_message: str) -> EmotionState:
        raw_score = self._score_agent_message(agent_message)

        # Map to delta based on quality:
        # Clear/helpful/removes doubt: reduce by 2 to 4 points
        # Partially helpful: reduce by 1 point
        # Confusing/incomplete/wrong: increase by 1 to 2 points
        # Very bad/ignores: increase by 2 to 3 points
        if raw_score >= 3:
            delta = -3 if self.state.intensity >= 4 else -2
        elif raw_score >= 1:
            delta = -1
        elif raw_score == 0:
            delta = 0
        elif raw_score == -1:
            delta = 1
        else:
            delta = 2

        old_intensity = self.state.intensity
        new_intensity = max(
            EMOTION_SCALE_MIN,
            min(EMOTION_SCALE_MAX, old_intensity + delta)
        )

        old_label = self.state.label
        new_label = EMOTION_LABELS[new_intensity]

        self.state.intensity = new_intensity
        self.state.label = new_label

        self.state.history.append(
            f"{old_label}({old_intensity}/5) → "
            f"{new_label}({new_intensity}/5) "
            f"[Δ{delta:+d}]"
        )

        return self.state

    def force_set(
        self,
        intensity: int,
        reason: str = "manual"
    ) -> EmotionState:
        intensity = max(
            EMOTION_SCALE_MIN,
            min(EMOTION_SCALE_MAX, intensity)
        )

        self.state.intensity = intensity
        self.state.label = EMOTION_LABELS[intensity]

        self.state.history.append(
            f"force → {intensity}/5 "
            f"({self.state.label}) [{reason}]"
        )

        return self.state