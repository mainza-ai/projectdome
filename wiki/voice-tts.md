# Voice Layer — Text-to-Speech

`src/voice/` — synthesizes speech audio from text. Fully local, permissively licensed.

## Provider Protocol

`src/voice/provider.py` — abstract interface:

```python
class VoiceProvider(Protocol):
    def synthesize(text: str) -> VoiceResult:
        """Returns audio (PCM float32) + sample_rate + phoneme_events"""
```

`PhonemeEvent` dataclass: `phoneme: str`, `start_time: float`, `end_time: float`
`VoiceResult` dataclass: `audio: np.ndarray`, `sample_rate: int`, `phoneme_events: List[PhonemeEvent]`

## Piper Provider

`src/voice/piper_provider.py` — implements VoiceProvider using Piper TTS.

- Model: `en_US-lessac-medium` (downloads from HuggingFace on first run)
- Auto-downloads `.onnx` and `.onnx.json` config to `data/voice/`
- Returns raw float32 audio at the model's native sample rate
- Phoneme events placeholder — populated later by forced alignment

```python
voice = PiperProvider()
result = voice.synthesize("Hello world")
# result.audio: np.ndarray, result.sample_rate: int
```

## File reference

| File | Role |
|---|---|
| `src/voice/__init__.py` | Re-exports VoiceProvider, VoiceResult, PhonemeEvent |
| `src/voice/provider.py` | Provider protocol + data classes |
| `src/voice/piper_provider.py` | Piper TTS implementation |

## Related

- [[acoustic-layer]] — high-level TTS comparison (Kokoro vs Piper)
- [[alignment-pipeline]] — forced alignment (populates phoneme_events)
