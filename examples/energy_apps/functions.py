"""Stage functions for the energy-profiling workloads.

Stages are compute-only (no FlexData in or out) and generate their data from a
seed, so they run on any backend without a storage bucket. Per-worker workload
size arrives through params.
"""

# ---------------------------------------------------------------------------
# Monte Carlo Pi: single stage, total work constant across worker counts.
# ---------------------------------------------------------------------------

TOTAL_POINTS = 100_000_000


def montecarlo_stage(ctx):
    """Count points inside the quarter circle."""
    import random

    n = ctx.get_param("points_per_worker")
    inside = 0
    for _ in range(n):
        x = random.random()
        y = random.random()
        if x * x + y * y <= 1.0:
            inside += 1
    return inside


# ---------------------------------------------------------------------------
# Titanic: single stage, fixed work per worker, so total work grows with W.
# ---------------------------------------------------------------------------

SAMPLES_PER_WORKER = 80_000


def titanic_stage(ctx):
    """Train a Random Forest on a synthetic chunk."""
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    seed = ctx.get_param("seed")
    n = ctx.get_param("samples")

    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 8))
    logit = X[:, 0] + 0.5 * X[:, 1] - 0.3 * X[:, 2] + rng.normal(scale=0.5, size=n)
    y = (logit > 0).astype(int)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed)
    # n_jobs=1 keeps intra-worker threading out of the measurement.
    clf = RandomForestClassifier(n_estimators=100, n_jobs=1, random_state=seed)
    clf.fit(Xtr, ytr)
    return float(clf.score(Xte, yte))


# ---------------------------------------------------------------------------
# ML ensemble: stage1 is parallel and dominant, the other three are serial.
# ---------------------------------------------------------------------------

ML_SAMPLES = 20_000
ML_FEATURES = 50
ML_COMPONENTS = 10


def _ml_data(seed, n=ML_SAMPLES, d=ML_FEATURES):
    import numpy as np

    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    w = rng.normal(size=d)
    y = ((X @ w + rng.normal(scale=0.5, size=n)) > 0).astype(int)
    return X, y


def ml_stage0_pca(ctx):
    """Dimensionality reduction."""
    from sklearn.decomposition import PCA

    seed = ctx.get_param("seed")
    X, _ = _ml_data(seed)
    pca = PCA(n_components=ML_COMPONENTS, random_state=seed)
    pca.fit(X)
    return float(pca.explained_variance_ratio_.sum())


def ml_stage1_train(ctx):
    """Train one model of the ensemble."""
    from sklearn.decomposition import PCA
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import train_test_split

    seed = ctx.get_param("seed")
    X, y = _ml_data(seed)
    Xr = PCA(n_components=ML_COMPONENTS, random_state=0).fit_transform(X)
    Xtr, Xte, ytr, yte = train_test_split(Xr, y, test_size=0.2, random_state=seed)
    clf = GradientBoostingClassifier(n_estimators=100, random_state=seed)
    clf.fit(Xtr, ytr)
    return float(clf.score(Xte, yte))


def ml_stage2_aggregate(ctx):
    """Combine the ensemble metrics."""
    return {"n_models": ctx.get_param("n_models")}


def ml_stage3_test(ctx):
    """Evaluate the ensemble on fresh data."""
    from sklearn.decomposition import PCA
    from sklearn.ensemble import GradientBoostingClassifier

    X, y = _ml_data(999)
    Xr = PCA(n_components=ML_COMPONENTS, random_state=0).fit_transform(X)
    clf = GradientBoostingClassifier(n_estimators=50, random_state=0).fit(Xr, y)
    return float(clf.score(Xr, y))


# ---------------------------------------------------------------------------
# Video pipeline: stage2 dominates; stage0 does not benefit from parallelism.
# ---------------------------------------------------------------------------

# Sized so the stages stay long enough to be measurable: the psutil monitor
# samples process CPU every 0.5s, so a stage under about 2.5s gets fewer than
# five samples and its utilisation figure is not a measurement.
TOTAL_FRAMES = 1200
FRAME_H = 240
FRAME_W = 320


def _frames(seed, count):
    """Yield frames one at a time. rng.normal returns float64, so batching the
    whole segment allocated ~2.2GB at 1200 frames before the cast to uint8."""
    import numpy as np

    rng = np.random.default_rng(seed)
    for _ in range(count):
        yield np.clip(
            rng.normal(128, 40, size=(FRAME_H, FRAME_W, 3)), 0, 255
        ).astype("uint8")


def video_stage0_segment(ctx):
    """Plan the frame segmentation."""
    workers = ctx.get_param("workers")
    per = TOTAL_FRAMES // workers
    return [(i, per if i < workers - 1 else TOTAL_FRAMES - per * (workers - 1))
            for i in range(workers)]


def video_stage1_extract(ctx):
    """Decode this worker's segment into frames."""
    seed = ctx.get_param("seed")
    count = ctx.get_param("frames_per_worker")
    return sum(1 for _ in _frames(seed, count))


def video_stage2_enhance(ctx):
    """Apply a 3x3 sharpen convolution to each frame."""
    import numpy as np

    seed = ctx.get_param("seed")
    count = ctx.get_param("frames_per_worker")
    k = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype="float32")
    out = 0.0
    for fr in _frames(seed, count):
        gray = fr.mean(axis=2)
        acc = np.zeros_like(gray)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                acc += k[dy + 1, dx + 1] * np.roll(np.roll(gray, dy, 0), dx, 1)
        out += float(np.abs(acc).mean())
    return out


def video_stage3_analyze(ctx):
    """Compute brightness and edge density per frame."""
    import numpy as np

    seed = ctx.get_param("seed")
    count = ctx.get_param("frames_per_worker")
    n = 0
    brightness_sum = 0.0
    edges = 0.0
    for fr in _frames(seed, count):
        n += 1
        brightness_sum += float(fr.mean())
        gray = fr.mean(axis=2)
        edges += float(np.abs(np.diff(gray, axis=1)).mean()
                       + np.abs(np.diff(gray, axis=0)).mean())
    # Frames are all the same size, so the mean of the per-frame means is the
    # overall mean.
    return {"brightness": brightness_sum / n if n else 0.0,
            "edges": edges / n if n else 0.0}
