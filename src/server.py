import os
import sys
import json
import base64
import urllib.parse
import io
import gc
import time
import re
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
from src.animation.emotion_blender import EmotionBlender
from gnm.shape.semantic_sampler import IdentitySampler, Gender, Ethnicity, Expression
from src.training.model import SpeechToCoefficientsModel

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
device = None

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

class GnmHTTPRequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def log_request(self, method, path, status, duration_ms):
        log.info(f"{method} {path} -> {status} ({duration_ms:.0f}ms)")

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
        t0 = time.time()
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
            content_type = content_types.get(ext, "text/plain")
            self.serve_file(local_path, content_type)
        except Exception as e:
            log.error(f"GET error: {e}")
            self.send_response(500)
            self.end_headers()
            self.wfile.write(b"Internal server error")
        self.log_request("GET", self.path, 200, (time.time() - t0) * 1000)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self):
        t0 = time.time()
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
            self.log_request("POST", self.path, 400, (time.time() - t0) * 1000)
            return

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        response = {}

        try:
            if self.path == "/api/speak":
                text = data.get("text", "")
                emotion = data.get("emotion", None)
                intensity = float(data.get("intensity", 1.0))
                style_id = int(data.get("style_id", 0))
                log.info(f"/api/speak: '{text[:60]}...' (emotion: {emotion}, style: {style_id})")
                audio, sr, visemes = pipeline.process(text)
                wav_io = io.BytesIO()
                sf.write(wav_io, audio, sr, format='WAV')
                wav_bytes = wav_io.getvalue()
                audio_base64 = base64.b64encode(wav_bytes).decode('utf-8')
                del wav_io, wav_bytes
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
                g_enum = Gender(gender)
                e_enum = Ethnicity(ethnicity)
                coeffs = identity_sampler.sample_identity(g_enum, e_enum, num_samples=1)[0]
                response = {"coefficients": coeffs.tolist()}

            elif self.path == "/api/identity/info":
                n = int(data.get("n", 10))
                names = [f"identity_{i:03d}" for i in range(min(n, 253))]
                response = {
                    "identity_dim": 253,
                    "component_names": names,
                    "num_components": min(n, 253),
                }

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
                    chunk_audio, chunk_sr, chunk_visemes = pipeline.process(sentence)
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
                    chunks.append({
                        "index": i,
                        "audio_base64": chunk_b64,
                        "duration": len(chunk_audio) / chunk_sr,
                    })
                wav_io = io.BytesIO()
                sf.write(wav_io, total_audio, total_sr, format='WAV')
                audio_base64 = base64.b64encode(wav_io.getvalue()).decode('utf-8')
                response = {
                    "audio_base64": audio_base64,
                    "visemes": [{"name": v.name, "start_time": v.start_time, "end_time": v.end_time} for v in all_visemes],
                    "chunks": chunks,
                    "num_sentences": len(sentences),
                }

            elif self.path == "/api/blink":
                left_wink = blender.sampler.sample_expression(Expression.WINK_LEFT, num_samples=1)[0]
                right_wink = blender.sampler.sample_expression(Expression.WINK_RIGHT, num_samples=1)[0]
                blink_coeffs = (left_wink + right_wink).tolist()
                response = {"coefficients": blink_coeffs}

            else:
                response = {"error": f"Unknown endpoint: {self.path}"}
                self.log_request("POST", self.path, 404, (time.time() - t0) * 1000)

        except Exception as e:
            log.error(f"/api{suffix_from_path(self.path)} failed: {traceback.format_exc()}")
            response = {"error": str(e)}
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self.wfile.write(json.dumps(response).encode('utf-8'))
        self.log_request("POST", self.path, 200, (time.time() - t0) * 1000)

def suffix_from_path(path):
    return path.replace("/api/", " ")

def run_server(port=8000):
    global pipeline, blender, identity_sampler
    log.info("=== Initializing Server Engines ===")
    pipeline = AcousticPipeline()
    blender = EmotionBlender()
    identity_sampler = IdentitySampler()
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
