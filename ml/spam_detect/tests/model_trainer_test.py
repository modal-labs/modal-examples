import pytest

from spam_detect import model_trainer


def dummy_classifier(email: str) -> float:
    _ = email
    return 0.86


def test_load_classifier_success(tmp_path):
    tmp_classifier_digest = model_trainer.store_classifier(
        classifier_func=dummy_classifier,
        classifier_destination_root=tmp_path,
        current_git_commit_hash="TEST-NOT-REALLY-A-COMMIT-HASH",
    )

    loaded_dummy_classifier = model_trainer.load_serialized_classifier(
        classifier_sha256_hash=tmp_classifier_digest,  # NOTE: Implementation detail leak
        classifier_destination_root=tmp_path,
    )
    test_email = "test email: doesn't matter what contents"
    assert dummy_classifier(email=test_email) == loaded_dummy_classifier(email=test_email)


def test_load_classifier_corrupted_data(tmp_path):
    dummy_classifier_b: bytes = model_trainer.serialize_classifier(dummy_classifier)
    # intentionally write classifier to non-content-addressed location. bogus path.
    bogus_hash = "bogus_pocus"
    bogus_path = tmp_path / bogus_hash
    with open(bogus_path, "wb") as f:
        f.write(dummy_classifier_b)

    with pytest.raises(ValueError):
        _ = model_trainer.load_serialized_classifier(
            classifier_sha256_hash=bogus_hash,
            classifier_destination_root=tmp_path,
        )
