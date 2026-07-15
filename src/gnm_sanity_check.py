import os
import numpy as np
from gnm.shape.gnm_numpy import GNM
from gnm.shape.data.versions.gnm_specs import GNMMajorVersion, GNMVariant
from gnm.shape.semantic_sampler import ExpressionSampler, Expression

def save_obj(vertices, triangles, filepath):
    """Save vertices and triangles as a Wavefront OBJ file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        # Write vertices
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        # Write faces (1-indexed)
        for t in triangles:
            f.write(f"f {t[0]+1} {t[1]+1} {t[2]+1}\n")
    print(f"Saved mesh to {filepath}")

def main():
    print("=== Loading GNM Head model ===")
    model = GNM.from_local(GNMMajorVersion.V3, GNMVariant.HEAD)
    print(f"Model loaded. Vertices: {model.num_vertices}, Joints: {model.num_joints}")

    # Generate identity, rotations, translation parameters (all zeros for neutral pose)
    identity = np.zeros(model.identity_dim)
    rotations = np.zeros((model.num_joints, 3))
    translation = np.zeros(3)

    # 1. Generate neutral mesh
    print("\n=== Generating Neutral Mesh ===")
    neutral_expr = np.zeros(model.expression_dim)
    neutral_vertices = model(identity, neutral_expr, rotations, translation)
    save_obj(neutral_vertices, model.triangles, "output/neutral.obj")

    # 2. Generate happy mesh using ExpressionSampler
    print("\n=== Generating Happy Mesh ===")
    sampler = ExpressionSampler()
    # sample_expression returns (1, 383)
    happy_expr = sampler.sample_expression(Expression.HAPPY).squeeze(0)
    happy_vertices = model(identity, happy_expr, rotations, translation)
    save_obj(happy_vertices, model.triangles, "output/happy.obj")

    # 3. Calculate max vertex displacement
    displacement = np.linalg.norm(happy_vertices - neutral_vertices, axis=-1)
    max_disp = np.max(displacement)
    mean_disp = np.mean(displacement)
    print(f"\n=== Displacement Sanity Check ===")
    print(f"Max vertex displacement: {max_disp:.6f} units")
    print(f"Mean vertex displacement: {mean_disp:.6f} units")
    
    if max_disp > 0.005:
        print("Displacement check PASSED! The mesh visibly deformed.")
    else:
        print("Displacement check FAILED! The mesh did not deform.")

if __name__ == "__main__":
    main()
