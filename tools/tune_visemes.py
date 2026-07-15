import os
import sys
import numpy as np

# Ensure project root is in python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.animation.viseme_table import VisemeTable, VISEMES
from src.animation.gnm_driver import GNMDriver

def print_help():
    print("\nCommands:")
    print("  list                  - List all available visemes")
    print("  show <viseme>         - Show non-zero coefficients for a viseme")
    print("  set <viseme> <idx> <val> - Set coefficient at index (0-181) to value")
    print("  export <viseme>       - Export viseme as OBJ to output/tune_<viseme>.obj")
    print("  save                  - Save changes to JSON table")
    print("  help                  - Print this help message")
    print("  exit                  - Exit the tuner")

def main():
    print("=== GNM Viseme Tuning Tool ===")
    table = VisemeTable()
    driver = GNMDriver()
    identity = np.zeros(driver.model.identity_dim, dtype=np.float32)

    print_help()

    while True:
        try:
            line = input("\ntune> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not line:
            continue

        parts = line.split()
        cmd = parts[0].lower()

        if cmd == "exit":
            break
        elif cmd == "help":
            print_help()
        elif cmd == "list":
            print("Available visemes:")
            for v in VISEMES:
                print(f" - {v}")
        elif cmd == "show":
            if len(parts) < 2:
                print("Usage: show <viseme>")
                continue
            name = parts[1]
            if name not in VISEMES:
                print(f"Unknown viseme: {name}")
                continue
            coeffs = table.get_coefficients(name)
            print(f"\nNon-zero coefficients for '{name}':")
            found = False
            for idx, val in enumerate(coeffs):
                if abs(val) > 1e-5:
                    label = driver.model.expression_names[idx + 200] if idx < 150 else f"tongue_{idx-150:03d}"
                    print(f"  Index {idx:3d} ({label:<25}): {val:8.4f}")
                    found = True
            if not found:
                print("  All coefficients are zero.")
        elif cmd == "set":
            if len(parts) < 4:
                print("Usage: set <viseme> <index> <value>")
                continue
            name, idx_str, val_str = parts[1], parts[2], parts[3]
            if name not in VISEMES:
                print(f"Unknown viseme: {name}")
                continue
            try:
                idx = int(idx_str)
                val = float(val_str)
                if not (0 <= idx < 182):
                    print("Index must be between 0 and 181")
                    continue
            except ValueError:
                print("Index and value must be numbers")
                continue
            
            coeffs = table.get_coefficients(name)
            coeffs[idx] = val
            table.set_coefficients(name, coeffs)
            label = driver.model.expression_names[idx + 200] if idx < 150 else f"tongue_{idx-150:03d}"
            print(f"Set '{name}' coefficient {idx} ({label}) to {val}")
        elif cmd == "export":
            if len(parts) < 2:
                print("Usage: export <viseme>")
                continue
            name = parts[1]
            if name not in VISEMES:
                print(f"Unknown viseme: {name}")
                continue
            coeffs = table.get_coefficients(name)
            
            # Map 182-dim viseme coeffs to full 383-dim GNM expression coeffs
            full_coeffs = np.zeros(383, dtype=np.float32)
            full_coeffs[200:382] = coeffs
            
            vertices = driver.evaluate(identity, full_coeffs)
            out_path = f"output/tune_{name}.obj"
            driver.save_mesh(vertices, out_path)
        elif cmd == "save":
            table.save()
        else:
            print(f"Unknown command: {cmd}")

if __name__ == "__main__":
    main()
