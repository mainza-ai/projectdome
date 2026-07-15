#!/usr/bin/env python3
"""Run all Project Dome unit tests."""
import subprocess
import sys
import os

test_dir = os.path.dirname(os.path.abspath(__file__))
tests = [
    "test_viseme_mapper.py",
    "test_viseme_table.py",
    "test_emotion_blender.py",
    "test_interpolator.py",
    "test_dataset_split.py",
    "test_gnm_forward.py",
]

exit_code = 0
for test in tests:
    path = os.path.join(test_dir, test)
    print(f"\n{'='*60}")
    print(f"Running {test}...")
    print(f"{'='*60}")
    result = subprocess.run([sys.executable, path], capture_output=False)
    if result.returncode != 0:
        print(f"  FAILED: {test} (exit code {result.returncode})")
        exit_code = 1

print(f"\n{'='*60}")
if exit_code == 0:
    print("All tests passed!")
else:
    print("Some tests failed.")
sys.exit(exit_code)
