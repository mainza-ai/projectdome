"""Server integration tests.

Requires the server to be running on localhost:8080.
Run with: ./venv/bin/python tests/test_server_endpoints.py
"""
import sys
import os
import json
import urllib.request
import base64
import io

BASE = "http://localhost:8080"

def req(path, data=None, json_response=True):
    if data is not None:
        body = json.dumps(data).encode('utf-8')
        r = urllib.request.Request(f"{BASE}{path}", data=body,
            headers={"Content-Type": "application/json"},
            method="POST")
    else:
        r = urllib.request.Request(f"{BASE}{path}")
    try:
        resp = urllib.request.urlopen(r, timeout=10)
        raw = resp.read().decode('utf-8')
        if json_response:
            return resp.status, json.loads(raw)
        return resp.status, raw
    except urllib.error.HTTPError as e:
        return e.code, {"error": str(e)}
    except urllib.error.URLError:
        return 0, {"error": "Server not running"}

def test_health():
    status, data = req("/api/health", {})
    assert status == 200, f"Health endpoint failed: {status}"
    assert data.get("status") == "ok"
    print("  PASS: health")

def test_emotion():
    status, data = req("/api/emotion", {"name": "HAPPY", "intensity": 1.0})
    assert status == 200
    assert len(data.get("coefficients", [])) == 383
    assert any(abs(v) > 0.001 for v in data["coefficients"]), "HAPPY should produce non-zero coeffs"
    print("  PASS: emotion_happy")

def test_neutral_emotion():
    status, data = req("/api/emotion", {"name": "", "intensity": 1.0})
    assert status == 200
    assert all(v == 0.0 for v in data["coefficients"]), "Empty emotion should produce zeros"
    print("  PASS: emotion_neutral")

def test_identity():
    status, data = req("/api/identity", {"gender": 0, "ethnicity": 2})
    assert status == 200
    assert len(data.get("coefficients", [])) == 253
    print("  PASS: identity")

def test_identity_info():
    status, data = req("/api/identity/info", {"n": 5})
    assert status == 200
    assert data["identity_dim"] == 253
    assert len(data["component_names"]) == 5
    print("  PASS: identity_info")

def test_blink():
    status, data = req("/api/blink", {})
    assert status == 200
    assert len(data.get("coefficients", [])) == 383
    print("  PASS: blink")

def test_speak():
    status, data = req("/api/speak", {"text": "Hello world."})
    assert status == 200
    assert len(data.get("audio_base64", "")) > 100
    assert len(data.get("visemes", [])) > 0
    # Verify audio is valid WAV
    wav_bytes = base64.b64decode(data["audio_base64"])
    assert wav_bytes[:4] == b'RIFF', "Audio should be valid WAV"
    print(f"  PASS: speak ({len(data['visemes'])} visemes, {len(wav_bytes)} bytes audio)")

def test_speak_stream():
    status, data = req("/api/speak/stream", {"text": "Hello. Welcome."})
    assert status == 200
    assert data["num_sentences"] >= 2
    assert len(data["chunks"]) >= 2
    assert len(data.get("audio_base64", "")) > 100
    print(f"  PASS: speak_stream ({data['num_sentences']} sentences, {len(data['chunks'])} chunks)")

def test_static_files():
    status, body = req("/index.html", json_response=False)
    assert status == 200, f"index.html returned {status}"
    assert '<!DOCTYPE html>' in body, "index.html should be HTML"
    status, body = req("/style.css", json_response=False)
    assert status == 200, f"style.css returned {status}"
    status, body = req("/renderer.js", json_response=False)
    assert status == 200, f"renderer.js returned {status}"
    status, body = req("/nonexistent.html", json_response=False)
    assert status == 404, f"nonexistent should 404, got {status}"
    print("  PASS: static_files")

if __name__ == "__main__":
    print("Server endpoint tests:")
    # Check server is running
    status, _ = req("/api/health", {})
    if status == 0:
        print("  SKIP: Server not running (start with ./venv/bin/python src/server.py)")
        sys.exit(0)
    
    test_health()
    test_emotion()
    test_neutral_emotion()
    test_identity()
    test_identity_info()
    test_blink()
    test_static_files()
    test_speak()
    test_speak_stream()
    print("\nAll server endpoint tests passed!")
