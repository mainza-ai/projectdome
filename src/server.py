import os
import sys
import json
import base64
import urllib.parse
import io
import soundfile as sf
from http.server import HTTPServer, BaseHTTPRequestHandler

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.alignment.pipeline import AcousticPipeline
from src.animation.emotion_blender import EmotionBlender
from gnm.shape.semantic_sampler import IdentitySampler, Gender, Ethnicity, Expression
import torch
import torchaudio
from src.training.model import SpeechToCoefficientsModel

# Global pipeline instances
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
            # We set MPS fallback flag to ensure linear interpolation op fallback to CPU behaves correctly
            import os
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
            
            device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
            print(f"[Server] Loading SpeechToCoefficients model on {device}...")
            
            checkpoint = torch.load(checkpoint_path, map_location=device)
            neural_model = SpeechToCoefficientsModel().to(device)
            neural_model.load_state_dict(checkpoint["model_state_dict"])
            neural_model.eval()
            print("[Server] SpeechToCoefficients model loaded successfully.")
        except Exception as e:
            print(f"[Server] Failed to load neural model: {e}")
            neural_model = None
    else:
        print("[Server] SpeechToCoefficients checkpoint not found. fallback to Path A.")

class GnmHTTPRequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Mute default request logs to prevent terminal spam
        pass

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
        url_path = urllib.parse.urlparse(self.path).path
        if url_path == "/" or url_path == "":
            url_path = "/index.html"

        # Serve from 'web/', 'data/', or 'assets/' folder
        if url_path.startswith("/data/") or url_path.startswith("/assets/"):
            local_path = url_path.lstrip("/")
        else:
            local_path = os.path.join("web", url_path.lstrip("/"))

        # Determine Content-Type
        if local_path.endswith(".html"):
            content_type = "text/html"
        elif local_path.endswith(".css"):
            content_type = "text/css"
        elif local_path.endswith(".js"):
            content_type = "application/javascript"
        elif local_path.endswith(".json"):
            content_type = "application/json"
        elif local_path.endswith(".bin"):
            content_type = "application/octet-stream"
        elif local_path.endswith(".wav"):
            content_type = "audio/wav"
        elif local_path.endswith(".png"):
            content_type = "image/png"
        elif local_path.endswith(".jpg") or local_path.endswith(".jpeg"):
            content_type = "image/jpeg"
        elif local_path.endswith(".gif"):
            content_type = "image/gif"
        else:
            content_type = "text/plain"

        self.serve_file(local_path, content_type)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        data = json.loads(post_data.decode('utf-8'))

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        response = {}

        if self.path == "/api/speak":
            text = data.get("text", "")
            emotion = data.get("emotion", None)
            intensity = data.get("intensity", 1.0)
            style_id = int(data.get("style_id", 0)) # style conditional conditioning index
            
            print(f"[Server] API /api/speak request: '{text}' (emotion: {emotion}, intensity: {intensity}, style: {style_id})")
            
            # Execute acoustic pipeline
            audio, sr, visemes = pipeline.process(text)
            
            # Encode audio to in-memory WAV buffer
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
            intensity = data.get("intensity", 1.0)
            print(f"[Server] API /api/emotion request: '{name}' (intensity: {intensity})")
            
            coeffs = blender.get_emotion_coefficients(name, intensity)
            response = {
                "coefficients": coeffs.tolist()
            }

        elif self.path == "/api/identity":
            gender = int(data.get("gender", 0))
            ethnicity = int(data.get("ethnicity", 2))
            print(f"[Server] API /api/identity request: gender={gender}, ethnicity={ethnicity}")
            
            g_enum = Gender(gender)
            e_enum = Ethnicity(ethnicity)
            
            # Generate 253 identity coefficients
            coeffs = identity_sampler.sample_identity(g_enum, e_enum, num_samples=1)[0]
            response = {
                "coefficients": coeffs.tolist()
            }

        elif self.path == "/api/blink":
            print("[Server] API /api/blink request")
            left_wink = blender.sampler.sample_expression(Expression.WINK_LEFT, num_samples=1)[0]
            right_wink = blender.sampler.sample_expression(Expression.WINK_RIGHT, num_samples=1)[0]
            # Blend left and right winks to create a full closed eyelid blink
            blink_coeffs = (left_wink + right_wink).tolist()
            response = {
                "coefficients": blink_coeffs
            }

        self.wfile.write(json.dumps(response).encode('utf-8'))

def run_server(port=8000):
    global pipeline, blender, identity_sampler
    print("=== Initializing Server Engines ===")
    pipeline = AcousticPipeline()
    blender = EmotionBlender()
    identity_sampler = IdentitySampler()
    print("=== Server Engines Initialized ===")

    server_address = ('', port)
    httpd = HTTPServer(server_address, GnmHTTPRequestHandler)
    print(f"=== Project Dome local web app is available at http://localhost:{port}/ ===")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
        httpd.server_close()

if __name__ == "__main__":
    run_server(port=8080)
