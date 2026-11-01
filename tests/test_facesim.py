import numpy as np
from PIL import Image

from facesim import (
    random_identity,
    render,
    Variation,
    generate_dataset,
    load_manifest,
    enroll_probe_split,
    make_pairs,
    run_verification,
    EmbeddingBackend,
    compute_metrics,
)


def test_identity_is_deterministic():
    a = random_identity(7)
    b = random_identity(7)
    c = random_identity(8)
    assert a == b
    assert a != c


def test_render_shape_and_type():
    p = random_identity(1)
    img = render(p, size=128)
    assert isinstance(img, Image.Image)
    assert img.size == (128, 128)
    assert img.mode == "RGB"


def test_variation_changes_pixels():
    p = random_identity(3)
    base = np.asarray(render(p, size=128, variation=Variation()))
    posed = np.asarray(render(p, size=128, variation=Variation(yaw=25, pitch=10)))
    assert base.shape == posed.shape
    assert not np.array_equal(base, posed)


def test_dataset_generation(tmp_path):
    samples = generate_dataset(tmp_path, n_identities=4, variations_per_identity=3,
                               size=96, seed=0)
    assert len(samples) == 12
    assert (tmp_path / "manifest.json").exists()
    reloaded = load_manifest(tmp_path)
    assert len(reloaded) == 12
    enroll, probe = enroll_probe_split(reloaded)
    assert len(enroll) == 4
    assert len(probe) == 8


def test_pairs_have_both_classes():
    samples = [s for s in generate_dataset_mem()]
    pairs = make_pairs(samples, max_impostor_per_genuine=1, seed=1)
    labels = {same for _, _, same in pairs}
    assert labels == {True, False}


def generate_dataset_mem():
    # Build Sample objects without touching disk (identity/variation only).
    from facesim.dataset import Sample
    out = []
    for ident in range(3):
        for vi in range(3):
            out.append(Sample(ident, vi, f"images/id{ident}_v{vi}.png", {}))
    return out


def test_metrics_perfect_separation():
    # genuine scores high, impostor scores low -> EER 0, AUC 1.
    scores = np.array([0.9, 0.95, 0.1, 0.05])
    labels = np.array([1, 1, 0, 0])
    m = compute_metrics(scores, labels)
    assert m.eer == 0.0
    assert m.auc == 1.0


def test_end_to_end_demo_backend(tmp_path):
    generate_dataset(tmp_path, n_identities=6, variations_per_identity=4,
                     size=128, seed=2)
    samples = load_manifest(tmp_path)
    pairs = make_pairs(samples, max_impostor_per_genuine=1, seed=2)
    m = run_verification(EmbeddingBackend(), pairs, root=tmp_path)
    # The demo baseline should do clearly better than chance on synthetic faces.
    assert m.auc > 0.75
    assert 0.0 <= m.eer <= 0.5
