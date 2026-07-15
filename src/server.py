import os
import sys
import json
import base64
import urllib.parse
import io
import gc
import time
import re
import signal
import logging
import traceback
import soundfile as sf
from http.server import HTTPServer, BaseHTTPRequestHandler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import numpy as np
import torch
from src.alignment.pipeline import AcousticPipeline
from src.animation.emotion_blender import EmotionBlender, _SilentIdentitySampler
from gnm.shape.semantic_sampler import Gender, Ethnicity, Expression
from src.training.model import SpeechToCoefficientsModel
from src.mind.local_provider import LocalMindProvider
from src.mind.conversation import ConversationContext

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger('dome')

pipeline = None
blender = None
identity_sampler = None
neural_model = None
mind_provider = None
conversation = None
device = None
REQUEST_TIMEOUT = 45

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Request timed out")

def init_neural_model():
    global neural_model, device
    checkpoint_path = "voca/model/checkpoints/best_model.pt"
    if os.path.exists(checkpoint_path):
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
            log.info(f"Loading SpeechToCoefficients model on {device}...")
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            neural_model = SpeechToCoefficientsModel().to(device)
            neural_model.load_state_dict(checkpoint["model_state_dict"])
            neural_model.eval()
            log.info("SpeechToCoefficients model loaded successfully.")
        except Exception as e:
            log.error(f"Failed to load neural model: {e}")
            neural_model = None
    else:
        log.info("SpeechToCoefficients checkpoint not found. fallback to Path A.")

def with_timeout(func):
    def wrapper(*args, **kwargs):
        if hasattr(signal, 'SIGALRM'):
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(REQUEST_TIMEOUT)
        try:
            return func(*args, **kwargs)
        except TimeoutError:
            raise
        finally:
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)
    return wrapper

class GnmHTTPRequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def log_request(self, code='-', size='-'):
        method = getattr(self, 'command', '?')
        path = getattr(self, 'path', '?')
        log.info(f"{method} {path} -> {code}")

    def serve_file(self, file_path, content_type):
        if not os.path.exists(file_path):
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"File not found")
            return
        file_size = os.path.getsize(file_path)
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(file_size))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        with open(file_path, 'rb') as f:
            self.wfile.write(f.read())

    def do_GET(self):
        try:
            url_path = urllib.parse.urlparse(self.path).path
            if url_path == "/" or url_path == "":
                url_path = "/index.html"
            if url_path.startswith("/data/") or url_path.startswith("/assets/"):
                local_path = url_path.lstrip("/")
            else:
                local_path = os.path.join("web", url_path.lstrip("/"))
            ext = os.path.splitext(local_path)[1].lower()
            content_types = {
                ".html": "text/html", ".css": "text/css", ".js": "application/javascript",
                ".json": "application/json", ".bin": "application/octet-stream",
                ".wav": "audio/wav", ".png": "image/png", ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg", ".gif": "image/gif", ".svg": "image/svg+xml",
                ".ico": "image/x-icon",
            }
            self.serve_file(local_path, content_types.get(ext, "text/plain"))
        except Exception as e:
            log.error(f"GET error: {e}")
            try:
                self.send_response(500)
                self.end_headers()
                self.wfile.write(b"Internal server error")
            except Exception:
                pass

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    @with_timeout
    def process_synthesis(self, text):
        return pipeline.process(text)

    def do_POST(self):
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length) if content_length > 0 else b'{}'
            data = json.loads(post_data.decode('utf-8')) if post_data else {}
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps({"error": f"Invalid JSON: {e}"}).encode('utf-8'))
            return

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        response = {}

        try:
            if self.path == "/api/chat":
                user_text = data.get("text", "")
                log.info(f"/api/chat: '{user_text[:60]}...'")
                mind_resp = mind_provider.generate(user_text, conversation.get_history())
                conversation.add_user_message(user_text)
                conversation.add_assistant_message(mind_resp.text)
                audio, sr, visemes = self.process_synthesis(mind_resp.text)
                wav_io = io.BytesIO()
                sf.write(wav_io, audio, sr, format='WAV')
                wav_bytes = wav_io.getvalue()
                audio_base64 = base64.b64encode(wav_bytes).decode('utf-8')
                if mind_resp.emotion:
                    emotion_coeffs = blender.get_emotion_coefficients(mind_resp.emotion, mind_resp.emotion_intensity)
                else:
                    emotion_coeffs = np.zeros(383, dtype=np.float32)
                response = {
                    "response_text": mind_resp.text,
                    "emotion": mind_resp.emotion,
                    "audio_base64": audio_base64,
                    "visemes": [{"name": v.name, "start_time": v.start_time, "end_time": v.end_time} for v in visemes],
                    "emotion_coefficients": emotion_coeffs.tolist(),
                }

            elif self.path == "/api/chat/reset":
                conversation.clear()
                response = {"status": "conversation reset"}

            elif self.path == "/api/speak":
                text = data.get("text", "")
                log.info(f"/api/speak: '{text[:60]}...'")
                audio, sr, visemes = self.process_synthesis(text)
                wav_io = io.BytesIO()
                sf.write(wav_io, audio, sr, format='WAV')
                wav_bytes = wav_io.getvalue()
                audio_base64 = base64.b64encode(wav_bytes).decode('utf-8')
                response = {
                    "audio_base64": audio_base64,
                    "visemes": [{"name": v.name, "start_time": v.start_time, "end_time": v.end_time} for v in visemes]
                }

            elif self.path == "/api/emotion":
                name = data.get("name", "")
                intensity = float(data.get("intensity", 1.0))
                coeffs = blender.get_emotion_coefficients(name, intensity)
                response = {"coefficients": coeffs.tolist()}

            elif self.path == "/api/identity":
                gender = int(data.get("gender", 0))
                ethnicity = int(data.get("ethnicity", 2))
                coeffs = identity_sampler.sample_identity(Gender(gender), Ethnicity(ethnicity), num_samples=1)[0]
                response = {"coefficients": coeffs.tolist()}

            elif self.path == "/api/identity/info":
                n = int(data.get("n", 10))
                names = [f"identity_{i:03d}" for i in range(min(n, 253))]
                response = {"identity_dim": 253, "component_names": names, "num_components": min(n, 253)}

            elif self.path == "/api/speak/stream":
                text = data.get("text", "")
                sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
                if not sentences:
                    sentences = [text]
                log.info(f"/api/speak/stream: {len(sentences)} sentence(s)")
                chunks = []
                total_audio = None
                total_sr = None
                all_visemes = []
                offset = 0.0
                for i, sentence in enumerate(sentences):
                    chunk_audio, chunk_sr, chunk_visemes = self.process_synthesis(sentence)
                    for v in chunk_visemes:
                        v.start_time += offset
                        v.end_time += offset
                    all_visemes.extend(chunk_visemes)
                    if total_audio is None:
                        total_audio = chunk_audio
                        total_sr = chunk_sr
                    else:
                        total_audio = np.concatenate([total_audio, chunk_audio])
                    offset = len(total_audio) / total_sr if total_sr else offset
                    wav_io = io.BytesIO()
                    sf.write(wav_io, chunk_audio, chunk_sr, format='WAV')
                    chunk_b64 = base64.b64encode(wav_io.getvalue()).decode('utf-8')
                    chunks.append({"index": i, "audio_base64": chunk_b64, "duration": len(chunk_audio) / chunk_sr})
                wav_io = io.BytesIO()
                sf.write(wav_io, total_audio, total_sr, format='WAV')
                audio_base64 = base64.b64encode(wav_io.getvalue()).decode('utf-8')
                response = {
                    "audio_base64": audio_base64,
                    "visemes": [{"name": v.name, "start_time": v.start_time, "end_time": v.end_time} for v in all_visemes],
                    "chunks": chunks, "num_sentences": len(sentences),
                }

            elif self.path == "/api/blink":
                left_wink = blender.sampler.sample_expression(Expression.WINK_LEFT, num_samples=1)[0]
                right_wink = blender.sampler.sample_expression(Expression.WINK_RIGHT, num_samples=1)[0]
                response = {"coefficients": ((left_wink + right_wink).tolist())}

            elif self.path == "/api/health":
                response = {
                    "status": "ok",
                    "pipeline": pipeline is not None,
                    "blender": blender is not None,
                    "identity_sampler": identity_sampler is not None,
                    "mind_provider": mind_provider is not None,
                }

            else:
                response = {"error": f"Unknown endpoint: {self.path}"}

        except TimeoutError:
            log.error(f"/api{self.path.replace('/api/',' ')} timed out after {REQUEST_TIMEOUT}s")
            response = {"error": f"Request timed out after {REQUEST_TIMEOUT}s"}
        except Exception as e:
            log.error(f"/api{self.path.replace('/api/',' ')} failed: {traceback.format_exc()}")
            response = {"error": str(e)}
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self.wfile.write(json.dumps(response).encode('utf-8'))

def run_server(port=8000):
    global pipeline, blender, identity_sampler, mind_provider, conversation
    log.info("=== Initializing Server Engines ===")
    pipeline = AcousticPipeline()
    blender = EmotionBlender()
    identity_sampler = _SilentIdentitySampler()
    mind_provider = LocalMindProvider()
    conversation = ConversationContext()
    log.info("=== Server Engines Initialized ===")
    server_address = ('', port)
    httpd = HTTPServer(server_address, GnmHTTPRequestHandler)
    log.info(f"=== Project Dome available at http://localhost:{port}/ ===")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        log.info("Stopping server.")
        httpd.server_close()

if __name__ == "__main__":
    run_server(port=8080)
