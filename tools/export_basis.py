import os
import sys
import json
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gnm.shape.gnm_numpy import GNM
from gnm.shape.data.versions.gnm_specs import GNMMajorVersion, GNMVariant

def main():
    print("=== Loading GNM model for Web Export ===")
    model = GNM.from_local(GNMMajorVersion.V3, GNMVariant.HEAD)
    print(f"Model loaded: {model.num_vertices} vertices, {model.num_joints} joints")

    out_dir = "data/web"
    os.makedirs(out_dir, exist_ok=True)

    mean_pos = model.template_vertex_positions.astype(np.float32)
    mean_pos.tofile(os.path.join(out_dir, "mean_positions.bin"))
    print(f"Saved mean positions ({mean_pos.nbytes / 1024:.1f} KB)")

    id_basis = model.vertex_identity_basis.astype(np.float16)
    id_basis.tofile(os.path.join(out_dir, "identity_basis.bin"))
    print(f"Saved identity basis ({id_basis.nbytes / 1024 / 1024:.1f} MB)")

    expr_basis = model.expression_basis.astype(np.float16)
    expr_basis.tofile(os.path.join(out_dir, "expression_basis.bin"))
    print(f"Saved expression basis ({expr_basis.nbytes / 1024 / 1024:.1f} MB)")

    faces = model.triangles.astype(np.uint32)
    faces.tofile(os.path.join(out_dir, "face_indices.bin"))
    print(f"Saved face indices ({faces.nbytes / 1024:.1f} KB)")

    weights = model.skinning_weights.astype(np.float32)
    weights.tofile(os.path.join(out_dir, "skinning_weights.bin"))
    print(f"Saved skinning weights ({weights.nbytes / 1024:.1f} KB)")

    regressor = model.joint_regressor.astype(np.float32)
    regressor.tofile(os.path.join(out_dir, "joint_regressor.bin"))
    print(f"Saved joint regressor ({regressor.nbytes / 1024:.1f} KB)")

    if hasattr(model, 'pose_correctives') and model.pose_correctives is not None:
        pc = model.pose_correctives.astype(np.float16)
        pc.tofile(os.path.join(out_dir, "pose_correctives.bin"))
        print(f"Saved pose correctives ({pc.nbytes / 1024 / 1024:.1f} MB)")
    else:
        print("Pose correctives not available in this GNM model version.")

    if hasattr(model, 'vertex_body_parts') and model.vertex_body_parts is not None:
        vp = model.vertex_body_parts.astype(np.int32)
        vp.tofile(os.path.join(out_dir, "vertex_body_parts.bin"))
        print(f"Saved vertex body parts ({vp.nbytes / 1024:.1f} KB)")
    else:
        print("Vertex body parts not available — generating from expression name heuristics.")
        vp = np.zeros(model.num_vertices, dtype=np.int32)
        skin_irises_mask = np.zeros(model.num_vertices, dtype=bool)
        if hasattr(model, 'vertex_expression_basis_norm'):
            norms = np.linalg.norm(model.expression_basis[:, :, :3].reshape(383, -1), axis=1)
            eye_expr_indices = list(range(0, 100))
            for ei in eye_expr_indices:
                activation = np.linalg.norm(model.expression_basis[ei], axis=1)
                skin_irises_mask |= (activation > activation.mean() + 2 * activation.std())
        vp.astype(np.int32).tofile(os.path.join(out_dir, "vertex_body_parts.bin"))
        print(f"Saved heuristic vertex body parts ({vp.nbytes / 1024:.1f} KB)")

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
        "template_joint_positions": model.template_joint_positions.tolist(),
        "has_pose_correctives": hasattr(model, 'pose_correctives') and model.pose_correctives is not None,
        "has_vertex_body_parts": hasattr(model, 'vertex_body_parts') and model.vertex_body_parts is not None,
    }
    with open(os.path.join(out_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    print("Saved metadata JSON")

    print("\n=== Export complete ===")
    total_mb = sum(
        os.path.getsize(os.path.join(out_dir, f)) for f in os.listdir(out_dir)
        if os.path.isfile(os.path.join(out_dir, f))
    ) / (1024 * 1024)
    print(f"Total export size: {total_mb:.1f} MB")

if __name__ == "__main__":
    main()
