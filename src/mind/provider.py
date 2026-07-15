from typing import Protocol, Optional
from dataclasses import dataclass

@dataclass
class MindResponse:
    text: str
    emotion: Optional[str] = None         # one of the 20 Expression enum keys (e.g. "HAPPY")
    emotion_intensity: float = 1.0       # 0.0 to 1.0

class MindProvider(Protocol):
    def generate(self, user_input: str, history: list) -> MindResponse:
        """Generate response text and intent emotion from user input and history."""
        ...
