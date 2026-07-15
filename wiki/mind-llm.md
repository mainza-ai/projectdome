# Mind Layer — LLM Cognitive Engine

`src/mind/` — the avatar's "brain." Processes user input, generates dialogue, and emits synchronized affect tags.

## Provider Protocol

`src/mind/provider.py` — abstract interface:

```python
class MindProvider(Protocol):
    def generate(user_input: str, history: list) -> MindResponse:
        """Returns text + optional emotion tag + intensity"""
```

`MindResponse` dataclass: `text: str`, `emotion: Optional[str]`, `emotion_intensity: float`

## Local LLM Provider

`src/mind/local_provider.py` — calls an OpenAI-compatible API at `http://127.0.0.1:9000/v1`.

System prompt instructs the LLM to return **JSON** with exactly two keys:
- `text` — spoken response
- `emotion` — one of 20 GNM ExpressionSampler labels (or `null` for neutral)

Valid emotions: `SURPRISE`, `DISGUST`, `HAPPY`, `SQUINT`, `SMILE_WIDE`, `CORNERS_DOWN`, `PUCKER`, `LIPS_ROLL_IN`, `SNARL`, `TONGUE_CENTER`, etc.

`response_format: {"type": "json_object"}` ensures parseable structured output.

On API failure, returns a graceful fallback response.

## Conversation Context

`src/mind/conversation.py` — maintains a sliding window of dialogue history.

- Default max history: 10 exchanges
- `add_user_message()` / `add_assistant_message()` — append to ring buffer
- `trim_history()` — keeps the last `max_history * 2` messages
- `clear()` — reset conversation

## File reference

| File | Role |
|---|---|
| `src/mind/__init__.py` | Re-exports MindProvider, MindResponse |
| `src/mind/provider.py` | Provider protocol definition |
| `src/mind/local_provider.py` | OpenAI-compatible local LLM client |
| `src/mind/conversation.py` | Conversation history ring buffer |
