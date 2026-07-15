import os
import sys
import glob
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger('training')

def check_vocaset_data():
    required = [
        "voca/trainingdata/templates.pkl",
        "voca/trainingdata/subj_seq_to_idx.pkl",
        "voca/trainingdata/raw_audio_fixed.pkl",
        "voca/trainingdata/data_verts.npy",
    ]
    missing = [f for f in required if not os.path.exists(f)]
    if missing:
        log.warning(f"VOCASET data missing: {missing}")
        log.warning("Download VOCASET from https://voca.is.tue.mpg.de and place in voca/trainingdata/")
        return False
    return True

def has_reprojected_data():
    files = glob.glob("voca/reprojected/*.npz")
    return len(files) > 0

def has_trained_checkpoint():
    return os.path.exists("voca/model/checkpoints/best_model.pt")

def run_reprojection():
    log.info("=== Running VOCASET reprojection ===")
    from src.training.reproject_vocaset import main as reproject
    reproject()
    log.info("=== Reprojection complete ===")

def run_training(epochs=30, batch_size=8, lr=1e-4):
    log.info(f"=== Starting training ({epochs} epochs, batch={batch_size}) ===")
    from src.training.train import train
    train(epochs=epochs, batch_size=batch_size, lr=lr)
    log.info("=== Training complete ===")

def run_evaluation():
    log.info("=== Running evaluation ===")
    from src.training.evaluate import evaluate
    evaluate()
    log.info("=== Evaluation complete ===")

def run_pipeline(epochs=30, batch_size=8, lr=1e-4, force_reproject=False):
    log.info("=" * 60)
    log.info("Project Dome Training Pipeline")
    log.info("=" * 60)
    if not check_vocaset_data():
        sys.exit(1)
    if force_reproject or not has_reprojected_data():
        run_reprojection()
    else:
        repro_count = len(glob.glob("voca/reprojected/*.npz"))
        log.info(f"Using existing reprojected data ({repro_count} files)")
    if not has_trained_checkpoint():
        log.info("No trained checkpoint found. Starting training...")
        run_training(epochs=epochs, batch_size=batch_size, lr=lr)
    else:
        log.info(f"Checkpoint found at voca/model/checkpoints/best_model.pt")
        retrain = os.environ.get("FORCE_RETRAIN", "").lower() in ("1", "true", "yes")
        if retrain:
            log.info("FORCE_RETRAIN set. Retraining...")
            run_training(epochs=epochs, batch_size=batch_size, lr=lr)
        else:
            log.info("Skipping training (set FORCE_RETRAIN=1 to retrain)")
    run_evaluation()
    log.info("=" * 60)
    log.info("Pipeline complete!")
    log.info("=" * 60)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Project Dome Training Pipeline")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--force-reproject", action="store_true", help="Force re-run reprojection")
    args = parser.parse_args()
    run_pipeline(
        epochs=args.epochs, batch_size=args.batch,
        lr=args.lr, force_reproject=args.force_reproject
    )
