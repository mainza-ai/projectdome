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
    print(f"Model: {model.num_vertices} vertices, {model.num_joints} joints, "
          f"{model.identity_dim} identity, {model.expression_dim} expression")

    out_dir = "data/web"
    os.makedirs(out_dir, exist_ok=True)

    mean_pos = model.template_vertex_positions.astype(np.float32)
    mean_pos.tofile(os.path.join(out_dir, "mean_positions.bin"))
    print(f"  mean_positions.bin: {mean_pos.nbytes/1024:.0f} KB")

    id_basis = model.vertex_identity_basis.astype(np.float16)
    id_basis.tofile(os.path.join(out_dir, "identity_basis.bin"))
    print(f"  identity_basis.bin: {id_basis.nbytes/1024/1024:.1f} MB")

    jid_basis = model.joint_identity_basis.astype(np.float16)
    jid_basis.tofile(os.path.join(out_dir, "joint_identity_basis.bin"))
    print(f"  joint_identity_basis.bin: {jid_basis.nbytes/1024:.0f} KB")

    expr_basis = model.expression_basis.astype(np.float16)
    expr_basis.tofile(os.path.join(out_dir, "expression_basis.bin"))
    print(f"  expression_basis.bin: {expr_basis.nbytes/1024/1024:.1f} MB")

    faces = model.triangles.astype(np.uint32)
    faces.tofile(os.path.join(out_dir, "face_indices.bin"))
    print(f"  face_indices.bin: {faces.nbytes/1024:.0f} KB")

    weights = model.skinning_weights.astype(np.float32)
    weights.tofile(os.path.join(out_dir, "skinning_weights.bin"))
    print(f"  skinning_weights.bin: {weights.nbytes/1024:.0f} KB")

    regressor = model.joint_regressor.astype(np.float32)
    regressor.tofile(os.path.join(out_dir, "joint_regressor.bin"))
    print(f"  joint_regressor.bin: {regressor.nbytes/1024:.0f} KB")

    pc = model.pose_correctives_regressor
    pc_nonzero = np.count_nonzero(pc)
    if pc is not None and pc_nonzero > 0:
        pc_file = os.path.join(out_dir, "pose_correctives_regressor.bin")
        pc.astype(np.float16).tofile(pc_file)
        print(f"  pose_correctives_regressor.bin: {pc.nbytes/1024/1024:.1f} MB (non-zero: {pc_nonzero})")
    else:
        print(f"  pose_correctives_regressor: all zeros (non-zero: {pc_nonzero}) — skipping export")

    body_parts = np.argmax(np.array(model.vertex_groups), axis=0).astype(np.int32)
    body_parts.tofile(os.path.join(out_dir, "vertex_body_parts.bin"))
    print(f"  vertex_body_parts.bin: {body_parts.nbytes/1024:.0f} KB ({len(model.vertex_group_names)} groups)")

    mirror = model.mirror_indices.astype(np.int32)
    mirror.tofile(os.path.join(out_dir, "mirror_indices.bin"))
    print(f"  mirror_indices.bin: {mirror.nbytes/1024:.0f} KB")

    tri_uvs = model.triangle_uvs.astype(np.float32)
    tri_uvs.tofile(os.path.join(out_dir, "triangle_uvs.bin"))
    print(f"  triangle_uvs.bin: {tri_uvs.nbytes/1024:.0f} KB")

    joint_angles_identity = np.zeros((model.identity_dim, model.num_joints, 3), dtype=np.float32)
    for i in range(model.identity_dim):
        for j in range(model.num_joints):
            joint_angles_identity[i, j] = model.joint_identity_basis[i, j]
    jai_flat = joint_angles_identity.astype(np.float16)
    jai_flat.tofile(os.path.join(out_dir, "joint_identity_angles.bin"))
    print(f"  joint_identity_angles.bin: {jai_flat.nbytes/1024:.0f} KB")

    metadata = {
        "num_vertices": model.num_vertices,
        "num_joints": model.num_joints,
        "identity_dim": model.identity_dim,
        "expression_dim": model.expression_dim,
        "num_triangles": len(model.triangles),
        "num_quads": len(model.quads),
        "expression_names": list(model.expression_names),
        "identity_names": list(model.identity_names),
        "joint_names": list(model.joint_names),
        "joint_parent_indices": [int(p) for p in model.joint_parent_indices],
        "template_joint_positions": np.array(model.template_joint_positions).tolist(),
        "group_names": list(model.vertex_group_names),
        "num_groups": len(model.vertex_group_names),
        "has_pose_correctives": pc is not None and pc_nonzero > 0,
        "has_joint_identity_basis": True,
        "has_mirror_indices": True,
        "has_triangle_uvs": True,
    }
    meta_path = os.path.join(out_dir, "metadata.json")
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  metadata.json: {os.path.getsize(meta_path)/1024:.0f} KB")

    total_mb = sum(
        os.path.getsize(os.path.join(out_dir, f)) for f in os.listdir(out_dir)
        if os.path.isfile(os.path.join(out_dir, f))
    ) / (1024 * 1024)
    print(f"\n=== Export complete: {total_mb:.1f} MB ===")

if __name__ == "__main__":
    main()
