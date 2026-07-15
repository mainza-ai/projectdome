from typing import List, Dict

class ConversationContext:
    def __init__(self, max_history: int = 10):
        self.history: List[Dict[str, str]] = []
        self.max_history = max_history

    def add_user_message(self, text: str):
        self.history.append({"role": "user", "content": text})
        self.trim_history()

    def add_assistant_message(self, text: str):
        self.history.append({"role": "assistant", "content": text})
        self.trim_history()

    def trim_history(self):
        # Keeps last N messages, keeping system prompt out of standard trim (handled in provider)
        if len(self.history) > self.max_history * 2:
            self.history = self.history[-(self.max_history * 2):]

    def get_history(self) -> List[Dict[str, str]]:
        return self.history

    def clear(self):
        self.history.clear()
