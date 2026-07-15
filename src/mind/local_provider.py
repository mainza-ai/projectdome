import json
import requests
from typing import List, Dict, Any, Optional
from src.mind.provider import MindProvider, MindResponse

EMOTIONS = [
    "SURPRISE", "DISGUST", "SUCK", "COMPRESS_FACE", "STRETCH_FACE", "HAPPY", 
    "SQUINT", "PLATYSMA", "BLOW", "FUNNELER", "SMILE_WIDE", "CORNERS_DOWN", 
    "PUCKER", "WINK_LEFT", "WINK_RIGHT", "MOUTH_LEFT", "MOUTH_RIGHT", 
    "LIPS_ROLL_IN", "SNARL", "TONGUE_CENTER"
]

class LocalMindProvider(MindProvider):
    def __init__(self, endpoint_url: str = "http://127.0.0.1:9000/v1"):
        self.endpoint_url = endpoint_url
        self.completions_url = f"{endpoint_url}/chat/completions"

    def generate(self, user_input: str, history: List[Dict[str, str]]) -> MindResponse:
        """Call local LLM server to generate a response text and emotion tag."""
        system_prompt = (
            "You are the brain of Project Dome — a conversational 3D avatar.\n"
            "You must respond in JSON format with exactly two keys:\n"
            "1. 'text': Your text response to the user.\n"
            "2. 'emotion': The emotion of your response. Choose EXACTLY one of the following uppercase words, or null for neutral:\n"
            f"   {', '.join(EMOTIONS)}\n\n"
            "Example JSON response:\n"
            '{"text": "I am so happy to see you!", "emotion": "HAPPY"}'
        )

        messages = [{"role": "system", "content": system_prompt}]
        for turn in history:
            messages.append(turn)
        messages.append({"role": "user", "content": user_input})

        payload = {
            "model": "local-model",
            "messages": messages,
            "response_format": {"type": "json_object"},
            "temperature": 0.7
        }

        try:
            response = requests.post(self.completions_url, json=payload, timeout=10)
            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                
                # Parse JSON output from LLM
                data = json.loads(content)
                text = data.get("text", "")
                emotion = data.get("emotion", None)
                
                # Validate emotion
                if emotion:
                    emotion = emotion.upper().strip()
                    if emotion not in EMOTIONS:
                        emotion = None
                        
                return MindResponse(text=text, emotion=emotion, emotion_intensity=1.0)
            else:
                print(f"[Mind] LLM request failed with status: {response.status_code}")
        except Exception as e:
            print(f"[Mind] Error communicating with local LLM: {e}")

        # Fallback response in case of API failure
        return MindResponse(
            text="I'm having trouble connecting to my brain right now.",
            emotion=None,
            emotion_intensity=0.0
        )
