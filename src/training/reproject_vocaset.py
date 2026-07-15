import os
import sys
import pickle
import numpy as np
from scipy.spatial import KDTree
from tqdm import tqdm

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from gnm.shape.gnm_numpy import GNM
from gnm.shape.data.versions.gnm_specs import GNMMajorVersion, GNMVariant
from gnm.shape.fitting_utils.project_on_pca import project_on_linear_vertex_basis

def compute_icp_alignment(src_mesh, tgt_mesh, max_iterations=50):
    """Compute optimal rotation R, translation t, and closest vertex indices mapping src to tgt."""
    src = np.copy(src_mesh)
    tgt = np.copy(tgt_mesh)

    # Initial centroid alignment
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
        
        # Apply transformation to current source
        current_src = (current_src @ R.T) + t
        
        # Accumulate transforms
        R_cum = R @ R_cum
        t_cum = R @ t_cum + t
        
        mean_err = np.mean(dists)
        if abs(prev_error - mean_err) < 1e-6:
            break
        prev_error = mean_err

    return R_cum, t_cum, indices

def main():
    print("=== Starting VOCASET Reprojection to GNM ===")
    
    # Load GNM
    model = GNM.from_local(GNMMajorVersion.V3, GNMVariant.HEAD)
    gnm_mean = model.template_vertex_positions
    gnm_basis = model.expression_basis
    
    # Load VOCASET paths
    voca_dir = "voca/trainingdata"
    templates_path = os.path.join(voca_dir, "templates.pkl")
    seq_to_idx_path = os.path.join(voca_dir, "subj_seq_to_idx.pkl")
    audio_path = os.path.join(voca_dir, "raw_audio_fixed.pkl")
    verts_path = os.path.join(voca_dir, "data_verts.npy")

    if not all(os.path.exists(p) for p in [templates_path, seq_to_idx_path, audio_path, verts_path]):
        print("Error: Missing VOCASET data files in voca/trainingdata/")
        sys.exit(1)

    print("Loading pickles...")
    voca_templates = pickle.load(open(templates_path, 'rb'), encoding='latin1')
    seq_to_idx = pickle.load(open(seq_to_idx_path, 'rb'), encoding='latin1')
    raw_audio = pickle.load(open(audio_path, 'rb'), encoding='latin1')
    
    print("Opening data_verts.npy (memory-mapped)...")
    data_verts = np.load(verts_path, mmap_mode='r')

    # Step 1: Pre-calculate ICP alignment for each speaker template to GNM template
    print("\nStep 1: Calculating speaker-specific ICP alignments to GNM...")
    speaker_alignments = {}
    for speaker, template in voca_templates.items():
        R, t, indices = compute_icp_alignment(template, gnm_mean)
        speaker_alignments[speaker] = {
            "R": R,
            "t": t,
            "indices": indices
        }
        print(f"  Aligned template for {speaker} to GNM (5023 -> 17821 vertices).")

    # Step 2: Reproject mesh frames
    out_dir = "voca/reprojected"
    os.makedirs(out_dir, exist_ok=True)
    
    print("\nStep 2: Processing sentence sequences...")
    for speaker in seq_to_idx.keys():
        align = speaker_alignments[speaker]
        R, t, speaker_indices = align["R"], align["t"], align["indices"]
        
        sentences = seq_to_idx[speaker].keys()
        print(f"Processing speaker: {speaker} ({len(sentences)} sentences)")
        
        for sentence in tqdm(sentences, desc=f"Reprojecting {speaker}"):
            # Get frame indices in the large data_verts array
            frame_map = seq_to_idx[speaker][sentence]
            frame_indices = [frame_map[i] for i in sorted(frame_map.keys())]
            
            # Read target meshes
            flame_meshes = data_verts[frame_indices] # shape (seq_len, 5023, 3)
            
            # Retrieve audio
            audio_item = raw_audio[speaker][sentence]
            audio_data = audio_item["audio"]
            sample_rate = audio_item["sample_rate"]
            
            # Run projection frame-by-frame
            seq_coeffs = []
            for frame_mesh in flame_meshes:
                # Align frame mesh to GNM space
                aligned_mesh = (frame_mesh @ R.T) + t
                
                # Project on GNM expression basis
                res = project_on_linear_vertex_basis(
                    aligned_mesh,
                    gnm_mean,
                    gnm_basis,
                    indices=speaker_indices,
                    regularization=1e-3, # slight regularization
                    compute_reconstruction=False
                )
                
                # res.coefficients has shape (1, 383)
                seq_coeffs.append(res.coefficients.squeeze(0))
                
            seq_coeffs = np.array(seq_coeffs, dtype=np.float32) # shape (seq_len, 383)
            
            # Extract 182 speech coefficients (lower face index 200-349 & tongue index 350-381)
            # Index 200:382 corresponds to GNM channels 200 to 381
            speech_coeffs = seq_coeffs[:, 200:382] # shape (seq_len, 182)
            
            # Save reprojected item
            out_file = os.path.join(out_dir, f"{speaker}_{sentence}.npz")
            np.savez_compressed(
                out_file,
                audio=audio_data,
                sample_rate=sample_rate,
                coefficients=speech_coeffs
            )

    print(f"\nReprojection complete! All reprojected NPZ files saved to: {out_dir}/")

if __name__ == "__main__":
    main()
