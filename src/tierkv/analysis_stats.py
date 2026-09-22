"""Paired-block statistics: means + 95% bootstrap CIs + 10% margin tests.

Frozen contract: blocks are the reps (n=5 per arm x length). Traces/tokens
are never reps. Warmup files are never pooled (callers filter warm=False).

Bootstrap: non-parametric percentile over block values (seeded, B=10000).
Paired contrasts: per-block differences within the same block index, then
bootstrap over the difference vector. With n=5, CIs are wide by construction;
callers must report width and flag underpowered contrasts.
"""
import random

B = 10000
SEED = 20260921


def _mean(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return sum(xs) / len(xs)


def boot_ci(xs, B=B, seed=SEED):
    """Mean + 95% percentile bootstrap CI. Returns dict with n, width."""
    xs = [x for x in xs if x is not None]
    n = len(xs)
    if n == 0:
        return {"mean": None, "lo": None, "hi": None, "n": 0,
                "width": None}
    mean = sum(xs) / n
    if n == 1:
        # n=1: single block value is the mean; there is no resampling
        # distribution, so the 95% CI is undefined (NOT zero-width).
        # Callers must treat lo/hi/width None as indeterminate and must
        # never interpret width 0.0 as a precise estimate.
        return {"mean": mean, "lo": None, "hi": None, "n": 1,
                "width": None}
    rng = random.Random(seed)
    stats = []
    for _ in range(B):
        s = [xs[rng.randrange(n)] for _ in range(n)]
        stats.append(sum(s) / n)
    stats.sort()
    lo = stats[int(0.025 * B)]
    hi = stats[min(B - 1, int(0.975 * B))]
    return {"mean": mean, "lo": lo, "hi": hi, "n": n, "width": hi - lo}


def paired_contrast(x_by_block, y_by_block, seed=SEED):
    """Paired per-block differences x-y over shared blocks + bootstrap CI."""
    shared = sorted(set(x_by_block) & set(y_by_block))
    dx = [x_by_block[b] for b in shared]
    dy = [y_by_block[b] for b in shared]
    diffs = [a - b for a, b in zip(dx, dy)
             if a is not None and b is not None]
    ci = boot_ci(diffs, seed=seed)
    ci["blocks"] = shared
    ci["n_blocks"] = len(shared)
    return ci


def margin_test(contrast, reference_mean, margin=0.10):
    """Test the 10% practical margin.

    Clears margin iff the whole 95% CI lies outside the +/-margin band
    around the reference (superiority beyond practical relevance), or the
    CI lies entirely inside the band (practical equivalence). Otherwise
    the contrast is indeterminate vs the margin (report as underpowered
    when the CI width itself exceeds the band width).
    Returns dict with verdict string + band.
    """
    if (contrast.get("mean") is None or reference_mean is None
            or reference_mean == 0):
        return {"verdict": "not_testable", "band": None}
    band = margin * abs(reference_mean)
    lo, hi, m = contrast["lo"], contrast["hi"], contrast["mean"]
    if lo is None:
        return {"verdict": "not_testable", "band": band}
    if lo > band or hi < -band:
        # CI fully beyond the band on one side: clears margin, direction kept
        direction = "positive" if m > 0 else "negative"
        return {"verdict": "clears_margin_%s" % direction, "band": band}
    if -band <= lo and hi <= band:
        return {"verdict": "practically_equivalent", "band": band}
    width = contrast.get("width")
    if width is not None and width > 2 * band:
        return {"verdict": "underpowered_wide_ci", "band": band}
    return {"verdict": "indeterminate_overlaps_band", "band": band}


def linfit(xys):
    """Ordinary least squares y ~ a + b*x. Returns a, b, r2, n."""
    pts = [(x, y) for x, y in xys if x is not None and y is not None]
    n = len(pts)
    if n < 2:
        return {"a": None, "b": None, "r2": None, "n": n}
    mx = sum(x for x, _ in pts) / n
    my = sum(y for _, y in pts) / n
    sxx = sum((x - mx) ** 2 for x, _ in pts)
    sxy = sum((x - mx) * (y - my) for x, y in pts)
    if sxx == 0:
        return {"a": my, "b": 0.0, "r2": 0.0, "n": n}
    b = sxy / sxx
    a = my - b * mx
    sst = sum((y - my) ** 2 for _, y in pts)
    sse = sum((y - (a + b * x)) ** 2 for x, y in pts)
    r2 = 1 - sse / sst if sst > 0 else 0.0
    return {"a": a, "b": b, "r2": r2, "n": n}
