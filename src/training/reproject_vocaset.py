import os
import sys
import pickle
import glob
import logging
import numpy as np
from scipy.spatial import KDTree
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from gnm.shape.gnm_numpy import GNM
from gnm.shape.data.versions.gnm_specs import GNMMajorVersion, GNMVariant
from gnm.shape.fitting_utils.project_on_pca import PCABasisProjection

log = logging.getLogger('training')

VOCA_SPEAKER_ORDER = [
    'FaceTalk_170728_03272_TA', 'FaceTalk_170904_00128_TA',
    'FaceTalk_170725_00137_TA', 'FaceTalk_170915_00223_TA',
    'FaceTalk_170811_03274_TA', 'FaceTalk_170913_03279_TA',
    'FaceTalk_170904_03276_TA', 'FaceTalk_170912_03278_TA',
    'FaceTalk_170811_03275_TA', 'FaceTalk_170908_03277_TA',
    'FaceTalk_170809_00138_TA', 'FaceTalk_170731_00024_TA',
]

def get_reprojected_count(out_dir):
    return len(glob.glob(os.path.join(out_dir, "*.npz")))

def compute_icp_alignment(src_mesh, tgt_mesh, max_iterations=50):
    src = np.copy(src_mesh)
    tgt = np.copy(tgt_mesh)
    t_src = src.mean(axis=0)
    t_tgt = tgt.mean(axis=0)
    current_src = src - t_src + t_tgt
    tree = KDTree(tgt)
    prev_error = 0
    indices = None
    R_cum = np.eye(3)
    t_cum = t_tgt - t_src
    for i in range(max_iterations):
        dists, indices = tree.query(current_src)
        matched = tgt[indices]
        c_src = current_src.mean(axis=0)
        c_tgt = matched.mean(axis=0)
        A = current_src - c_src
        B = matched - c_tgt
        H = A.T @ B
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[2, :] *= -1
            R = Vt.T @ U.T
        t = c_tgt - R @ c_src
        current_src = (current_src @ R.T) + t
        R_cum = R @ R_cum
        t_cum = R @ t_cum + t
        mean_err = np.mean(dists)
        if abs(prev_error - mean_err) < 1e-6:
            break
        prev_error = mean_err
    return R_cum, t_cum, indices

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
    log.info("=== Starting VOCASET Reprojection to GNM ===")
    model = GNM.from_local(GNMMajorVersion.V3, GNMVariant.HEAD)
    gnm_mean = model.template_vertex_positions
    gnm_basis = model.expression_basis
    voca_dir = "voca/trainingdata"
    templates_path = os.path.join(voca_dir, "templates.pkl")
    seq_to_idx_path = os.path.join(voca_dir, "subj_seq_to_idx.pkl")
    audio_path = os.path.join(voca_dir, "raw_audio_fixed.pkl")
    verts_path = os.path.join(voca_dir, "data_verts.npy")
    if not all(os.path.exists(p) for p in [templates_path, seq_to_idx_path, audio_path, verts_path]):
        log.error("Missing VOCASET data files in voca/trainingdata/")
        sys.exit(1)
    out_dir = "voca/reprojected"
    os.makedirs(out_dir, exist_ok=True)
    existing_count = get_reprojected_count(out_dir)
    if existing_count > 0:
        log.info(f"Found {existing_count} existing reprojected files. Skipping already-completed sentences.")
    else:
        log.info("No existing reprojected files found. Starting fresh.")
    log.info("Loading pickles...")
    voca_templates = pickle.load(open(templates_path, 'rb'), encoding='latin1')
    seq_to_idx = pickle.load(open(seq_to_idx_path, 'rb'), encoding='latin1')
    raw_audio = pickle.load(open(audio_path, 'rb'), encoding='latin1')
    log.info("Opening data_verts.npy (memory-mapped)...")
    data_verts = np.load(verts_path, mmap_mode='r')
    log.info("Step 1: Calculating speaker-specific ICP alignments to GNM...")
    speaker_alignments = {}
    for speaker, template in voca_templates.items():
        R, t, indices = compute_icp_alignment(template, gnm_mean)
        speaker_alignments[speaker] = {"R": R, "t": t, "indices": indices}
        log.info(f"  Aligned template for {speaker} to GNM (5023 -> 17821 vertices).")
    log.info("Step 2: Processing sentence sequences...")
    total_skipped = 0
    total_processed = 0
    for speaker in seq_to_idx.keys():
        align = speaker_alignments[speaker]
        R, t, speaker_indices = align["R"], align["t"], align["indices"]
        projector = PCABasisProjection(
            mean_vertex_positions=gnm_mean[speaker_indices],
            vertex_basis=gnm_basis[:, speaker_indices],
            vertex_indices=None,
            regularization=1e-3,
            compute_reconstruction=False
        )
        sentences = seq_to_idx[speaker].keys()
        log.info(f"Processing speaker: {speaker} ({len(sentences)} sentences)")
        for sentence in tqdm(sentences, desc=f"Reprojecting {speaker}"):
            out_file = os.path.join(out_dir, f"{speaker}_{sentence}.npz")
            if os.path.exists(out_file):
                total_skipped += 1
                continue
            frame_map = seq_to_idx[speaker][sentence]
            frame_indices = [frame_map[i] for i in sorted(frame_map.keys())]
            flame_meshes = data_verts[frame_indices]
            if sentence not in raw_audio.get(speaker, {}):
                log.warning(f"Skipping {speaker} {sentence} — audio missing.")
                continue
            audio_item = raw_audio[speaker][sentence]
            audio_data = audio_item["audio"]
            sample_rate = audio_item["sample_rate"]
            aligned_meshes = (flame_meshes @ R.T) + t
            res = projector(aligned_meshes)
            seq_coeffs = res.coefficients.astype(np.float32)
            speech_coeffs = seq_coeffs[:, 200:382]
            try:
                speaker_id = VOCA_SPEAKER_ORDER.index(speaker)
            except ValueError:
                speaker_id = -1
            np.savez_compressed(
                out_file, audio=audio_data, sample_rate=sample_rate,
                coefficients=speech_coeffs, speaker_id=speaker_id,
            )
            total_processed += 1
    log.info(f"Reprojection complete! Processed: {total_processed}, Skipped: {total_skipped}")
    log.info(f"Total files in {out_dir}/: {get_reprojected_count(out_dir)}")

if __name__ == "__main__":
    main()
