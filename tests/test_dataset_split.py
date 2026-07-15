import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.training.dataset import (
    VocasetDataset, VOCA_TRAIN_SUBJECTS, VOCA_VAL_SUBJECTS,
    VOCA_TEST_SUBJECTS, VOCA_ALL_SUBJECTS
)

def test_subject_splits_are_disjoint():
    train_set = set(VOCA_TRAIN_SUBJECTS)
    val_set = set(VOCA_VAL_SUBJECTS)
    test_set = set(VOCA_TEST_SUBJECTS)
    assert train_set.isdisjoint(val_set), "Train and val subjects overlap!"
    assert train_set.isdisjoint(test_set), "Train and test subjects overlap!"
    assert val_set.isdisjoint(test_set), "Val and test subjects overlap!"
    all_union = train_set | val_set | test_set
    assert len(all_union) == len(VOCA_ALL_SUBJECTS), "Not all subjects are covered by splits"
    assert len(all_union) == 12, f"Expected 12 total subjects, got {len(all_union)}"
    print(f"  PASS: subject_splits_are_disjoint ({len(train_set)} train, {len(val_set)} val, {len(test_set)} test)")

def test_voca_subject_order():
    expected_order = [
        'FaceTalk_170728_03272_TA', 'FaceTalk_170904_00128_TA',
        'FaceTalk_170725_00137_TA', 'FaceTalk_170915_00223_TA',
        'FaceTalk_170811_03274_TA', 'FaceTalk_170913_03279_TA',
        'FaceTalk_170904_03276_TA', 'FaceTalk_170912_03278_TA',
        'FaceTalk_170811_03275_TA', 'FaceTalk_170908_03277_TA',
        'FaceTalk_170809_00138_TA', 'FaceTalk_170731_00024_TA',
    ]
    assert VOCA_ALL_SUBJECTS == expected_order, "VOCA speaker order does not match expected"
    print("  PASS: voca_subject_order")

def test_train_subjects_match_voca_paper():
    assert len(VOCA_TRAIN_SUBJECTS) == 8, "VOCA paper uses 8 train subjects"
    assert VOCA_VAL_SUBJECTS == ['FaceTalk_170811_03275_TA', 'FaceTalk_170908_03277_TA']
    assert VOCA_TEST_SUBJECTS == ['FaceTalk_170809_00138_TA', 'FaceTalk_170731_00024_TA']
    print("  PASS: train_subjects_match_voca_paper")

def test_speaker_id_indices():
    for i, speaker in enumerate(VOCA_ALL_SUBJECTS):
        assert VOCA_ALL_SUBJECTS.index(speaker) == i
    # Train subjects are indices 0-7
    for i, speaker in enumerate(VOCA_TRAIN_SUBJECTS):
        assert VOCA_ALL_SUBJECTS.index(speaker) == i
    # Val subjects are indices 8-9
    for i, speaker in enumerate(VOCA_VAL_SUBJECTS):
        assert VOCA_ALL_SUBJECTS.index(speaker) == i + 8
    # Test subjects are indices 10-11
    for i, speaker in enumerate(VOCA_TEST_SUBJECTS):
        assert VOCA_ALL_SUBJECTS.index(speaker) == i + 10
    print("  PASS: speaker_id_indices")

if __name__ == "__main__":
    print("Dataset split tests:")
    test_subject_splits_are_disjoint()
    test_voca_subject_order()
    test_train_subjects_match_voca_paper()
    test_speaker_id_indices()
    print("\nAll dataset split tests passed!")
