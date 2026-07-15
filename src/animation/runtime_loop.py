import os
import numpy as np
from typing import Optional
from src.alignment.pipeline import AcousticPipeline
from src.animation.viseme_table import VisemeTable
from src.animation.interpolator import VisemeInterpolator
from src.animation.emotion_blender import EmotionBlender
from src.animation.gnm_driver import GNMDriver

def animate_utterance(
    text: str,
    emotion: Optional[str] = None,
    intensity: float = 1.0,
    fps: int = 30,
    output_dir: str = "output/frames"
):
    """Full offline pipeline: text -> audio + visemes -> GNM mesh frames -> OBJ files."""
    # 1. Process text to get audio and viseme timeline
    print("\n--- Running Acoustic Pipeline ---")
    pipeline = AcousticPipeline()
    audio, sr, visemes = pipeline.process(text)
    
    # Calculate duration
    audio_duration = len(audio) / sr
    total_frames = int(audio_duration * fps)
    print(f"Audio Duration: {audio_duration:.2f}s, Target Frames at {fps} fps: {total_frames}")

    # 2. Setup animation components
    print("\n--- Setting up Animation Components ---")
    viseme_table = VisemeTable()
    interpolator = VisemeInterpolator(viseme_table)
    blender = EmotionBlender()
    driver = GNMDriver()

    # Pre-sample emotion coefficients
    emotion_coeffs = blender.get_emotion_coefficients(emotion, intensity)

    # 3. Clean target directory
    os.makedirs(output_dir, exist_ok=True)
    for f in os.listdir(output_dir):
        if f.endswith(".obj"):
            try:
                os.remove(os.path.join(output_dir, f))
            except OSError:
                pass

    print(f"\n--- Generating {total_frames} Frames ---")
    # Neutral identity coefficients (zeros)
    identity_coeffs = np.zeros(driver.model.identity_dim, dtype=np.float32)

    for i in range(total_frames):
        time_s = i / fps
        
        # Look up and interpolate speech viseme shape (182-dim)
        speech_coeffs = interpolator.get_coefficients(time_s, visemes)
        
        # Additive blend speech visemes + non-speech emotion (383-dim)
        merged_coeffs = blender.blend(speech_coeffs, emotion_coeffs)
        
        # Compute 3D vertices
        vertices = driver.evaluate(identity_coeffs, merged_coeffs)
        
        # Save to OBJ file
        frame_path = os.path.join(output_dir, f"frame_{i:04d}.obj")
        driver.save_mesh(vertices, frame_path)
        
        if i % 10 == 0 or i == total_frames - 1:
            print(f"Rendered frame {i}/{total_frames} (time: {time_s:.2f}s)")

    print(f"\nSuccessfully rendered all frames to {output_dir}/")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate 3D GNM avatar frames from text and emotion.")
    parser.add_argument("--text", type=str, default="Welcome to Project Dome.", help="Text to speak")
    parser.add_argument("--emotion", type=str, default="HAPPY", help="Emotion label (e.g. HAPPY, SURPRISE)")
    parser.add_argument("--intensity", type=float, default=1.0, help="Intensity of the emotion (0.0 to 1.0)")
    parser.add_argument("--fps", type=int, default=30, help="Frame rate")
    parser.add_argument("--out", type=str, default="output/frames", help="Output folder path")
    args = parser.parse_args()

    animate_utterance(
        text=args.text,
        emotion=args.emotion,
        intensity=args.intensity,
        fps=args.fps,
        output_dir=args.out
    )
