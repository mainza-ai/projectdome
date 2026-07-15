import os
import sys
import json
import numpy as np

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gnm.shape.gnm_numpy import GNM
from gnm.shape.data.versions.gnm_specs import GNMMajorVersion, GNMVariant

def main():
    print("=== Loading GNM model for Web Export ===")
    model = GNM.from_local(GNMMajorVersion.V3, GNMVariant.HEAD)
    print("Model loaded.")

    out_dir = "data/web"
    os.makedirs(out_dir, exist_ok=True)

    # 1. Export template vertex positions (float32) -> mean_positions.bin
    mean_pos = model.template_vertex_positions.astype(np.float32)
    mean_pos_path = os.path.join(out_dir, "mean_positions.bin")
    mean_pos.tofile(mean_pos_path)
    print(f"Saved mean positions to {mean_pos_path} ({mean_pos.nbytes / 1024:.2f} KB)")

    # 2. Export vertex identity basis (float16) -> identity_basis.bin
    id_basis = model.vertex_identity_basis.astype(np.float16)
    id_basis_path = os.path.join(out_dir, "identity_basis.bin")
    id_basis.tofile(id_basis_path)
    print(f"Saved identity basis to {id_basis_path} ({id_basis.nbytes / 1024 / 1024:.2f} MB)")

    # 3. Export expression basis (float16) -> expression_basis.bin
    expr_basis = model.expression_basis.astype(np.float16)
    expr_basis_path = os.path.join(out_dir, "expression_basis.bin")
    expr_basis.tofile(expr_basis_path)
    print(f"Saved expression basis to {expr_basis_path} ({expr_basis.nbytes / 1024 / 1024:.2f} MB)")

    # 4. Export face indices (uint32) -> face_indices.bin
    faces = model.triangles.astype(np.uint32)
    faces_path = os.path.join(out_dir, "face_indices.bin")
    faces.tofile(faces_path)
    print(f"Saved face indices to {faces_path} ({faces.nbytes / 1024:.2f} KB)")

    # 5. Export skinning weights (float32) -> skinning_weights.bin
    weights = model.skinning_weights.astype(np.float32) # shape (4, 17821)
    weights_path = os.path.join(out_dir, "skinning_weights.bin")
    weights.tofile(weights_path)
    print(f"Saved skinning weights to {weights_path} ({weights.nbytes / 1024:.2f} KB)")

    # 6. Export joint regressor (float32) -> joint_regressor.bin
    regressor = model.joint_regressor.astype(np.float32) # shape (4, 17821)
    regressor_path = os.path.join(out_dir, "joint_regressor.bin")
    regressor.tofile(regressor_path)
    print(f"Saved joint regressor to {regressor_path} ({regressor.nbytes / 1024:.2f} KB)")

    # 7. Export metadata JSON
    metadata = {
        "num_vertices": model.num_vertices,
        "num_joints": model.num_joints,
        "identity_dim": model.identity_dim,
        "expression_dim": model.expression_dim,
        "num_triangles": len(model.triangles),
        "expression_names": model.expression_names,
        "identity_names": model.identity_names,
        "joint_names": model.joint_names,
        "joint_parent_indices": [int(p) for p in model.joint_parent_indices],
        "template_joint_positions": model.template_joint_positions.tolist()
    }
    metadata_path = os.path.join(out_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata JSON to {metadata_path}")

if __name__ == "__main__":
    main()
