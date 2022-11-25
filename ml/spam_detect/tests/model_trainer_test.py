import pytest

from spam_detect import model_trainer


def dummy_classifier(email: str) -> float:
    _ = email
    return 0.86


def test_hashtag_from_bytes():
    b = b"1234"
    expected = "sha256.03AC674216F3E15C761EE1A5E255F067953623C8B388B4459E13F978D7C846F4"
    assert model_trainer.create_hashtag_from_bytes(b) == expected


def test_hashtag_from_dir(tmp_path):
    dir = tmp_path / "dir1"
    dir.mkdir()
    contents = {
        "one": b"1234",
        "two": b"5678",
        "three": b"hello world",
    }
    for filename, b in contents.items():
        p = dir / filename
        p.write_bytes(b)

    hashtag = model_trainer.create_hashtag_from_dir(dir)
    assert hashtag == "sha256.EC0DAE99CA050AAC0E8D01A54997BB2BFA00E251990AE0A9239AE20670E86CEC"
    # If we add file, the hash changes
    p = dir / "four"
    p.write_bytes(b"I change the hash")
    hashtag_2 = model_trainer.create_hashtag_from_dir(dir)
    assert hashtag != hashtag_2
    # Renaming a file changes the hash
    p = dir / "one"
    p.rename(dir / "one.2")
    hashtag_3 = model_trainer.create_hashtag_from_dir(dir)
    assert hashtag != hashtag_3
    assert hashtag_2 != hashtag_3


def test_load_classifier_success(tmp_path):
    tmp_classifier_digest = model_trainer.store_classifier(
        classifier_func=dummy_classifier,
        classifier_destination_root=tmp_path,
        current_git_commit_hash="TEST-NOT-REALLY-A-COMMIT-HASH",
    )

    loaded_dummy_classifier = model_trainer.load_serialized_classifier(
        classifier_sha256_hash=tmp_classifier_digest,
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
