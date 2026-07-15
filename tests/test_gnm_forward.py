"""GNM forward pass regression tests.

Verifies that our export pipeline produces buffers that reproduce
the GNM NumPy model's output with high precision.
"""
import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gnm.shape.gnm_numpy import GNM
from gnm.shape.data.versions.gnm_specs import GNMMajorVersion, GNMVariant

def load_exported_buffers(data_dir="data/web"):
    buffers = {}
    for name in ['mean_positions', 'identity_basis', 'expression_basis',
                  'skinning_weights', 'joint_regressor']:
        path = os.path.join(data_dir, f"{name}.bin")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing buffer: {path}")
        buffers[name] = np.fromfile(path, dtype=np.float32)
    
    # Float16 buffers
    for name in ['identity_basis', 'expression_basis']:
        raw = np.fromfile(os.path.join(data_dir, f"{name}.bin"), dtype=np.float16)
        buffers[name] = raw.astype(np.float32)
    
    # Try joint_identity_basis
    jid_path = os.path.join(data_dir, "joint_identity_basis.bin")
    if os.path.exists(jid_path):
        buffers['joint_identity_basis'] = np.fromfile(jid_path, dtype=np.float16).astype(np.float32)
    
    with open(os.path.join(data_dir, "metadata.json")) as f:
        buffers['metadata'] = json.load(f)
    
    return buffers

def test_template_vertices():
    """Exported mean_positions should match GNM template_vertex_positions."""
    model = GNM.from_local(GNMMajorVersion.V3, GNMVariant.HEAD)
    exported = np.fromfile("data/web/mean_positions.bin", dtype=np.float32)
    expected = np.array(model.template_vertex_positions).flatten().astype(np.float32)
    assert len(exported) == len(expected), f"Length mismatch: {len(exported)} vs {len(expected)}"
    max_diff = np.max(np.abs(exported - expected))
    assert max_diff < 1e-5, f"Template verts max diff: {max_diff}"
    print(f"  PASS: template_vertices (max_diff={max_diff:.2e})")

def test_identity_basis():
    """Exported identity basis should match GNM."""
    model = GNM.from_local(GNMMajorVersion.V3, GNMVariant.HEAD)
    exported  = np.fromfile("data/web/identity_basis.bin", dtype=np.float16).astype(np.float32)
    expected = np.array(model.vertex_identity_basis).flatten().astype(np.float32)
    assert len(exported) == len(expected)
    max_diff = np.max(np.abs(exported - expected))
    assert max_diff < 0.001, f"Identity basis max diff: {max_diff}"
    print(f"  PASS: identity_basis (max_diff={max_diff:.2e})")

def test_forward_pass():
    """Exported buffers should produce same output as GNM forward pass."""
    model = GNM.from_local(GNMMajorVersion.V3, GNMVariant.HEAD)
    rng = np.random.default_rng(42)
    identity = rng.normal(0, 0.5, size=model.identity_dim).astype(np.float32)
    expression = rng.normal(0, 0.5, size=model.expression_dim).astype(np.float32)
    rotations = np.zeros((model.num_joints, 3), dtype=np.float32)
    translation = np.zeros(3, dtype=np.float32)
    
    # NumPy forward pass
    expected = np.array(model(identity, expression, rotations, translation))
    
    # Manual forward pass using exported buffers
    mean = np.fromfile("data/web/mean_positions.bin", dtype=np.float32).reshape(-1, 3)
    id_basis = np.fromfile("data/web/identity_basis.bin", dtype=np.float16).astype(np.float32).reshape(model.identity_dim, -1, 3)
    ex_basis = np.fromfile("data/web/expression_basis.bin", dtype=np.float16).astype(np.float32).reshape(model.expression_dim, -1, 3)
    
    bind_pose = mean.copy()
    for i in range(model.identity_dim):
        if abs(identity[i]) > 1e-4:
            bind_pose += id_basis[i] * identity[i]
    for i in range(model.expression_dim):
        if abs(expression[i]) > 1e-4:
            bind_pose += ex_basis[i] * expression[i]
    
    max_diff = np.max(np.abs(bind_pose - expected))
    assert max_diff < 0.005, f"Forward pass max diff: {max_diff}"
    print(f"  PASS: forward_pass (max_diff={max_diff:.2e})")

def test_vertex_count():
    """All buffers should have consistent vertex counts."""
    model = GNM.from_local(GNMMajorVersion.V3, GNMVariant.HEAD)
    n = model.num_vertices
    metadata = json.load(open("data/web/metadata.json"))
    assert metadata["num_vertices"] == n
    mean = np.fromfile("data/web/mean_positions.bin", dtype=np.float32)
    assert mean.shape[0] == n * 3
    faces = np.fromfile("data/web/face_indices.bin", dtype=np.uint32)
    assert faces.shape[0] % 3 == 0
    print(f"  PASS: vertex_count (n={n}, faces={faces.shape[0]//3})")

if __name__ == "__main__":
    print("GNM forward pass tests:")
    test_template_vertices()
    test_identity_basis()
    test_forward_pass()
    test_vertex_count()
    print("\nAll GNM forward pass tests passed!")
