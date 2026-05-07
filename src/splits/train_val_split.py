import numpy as np


def create_train_val_split(
    n_intervals: int,
    T: int,
    m: int,
    p: int = 10,
    q_factor: float = 1.5,
    seed: int = 42,
    Cm: np.ndarray = None,
    labels: np.ndarray = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if n_intervals < T + m:
        raise ValueError(
            f"Not enough intervals ({n_intervals}) for T={T}, m={m}")

    q = max(int(T * q_factor), T + 1)
    max_start = n_intervals - q
    if max_start <= 0:
        q = max(T + 1, n_intervals // 2)
        max_start = n_intervals - q

    rng = np.random.default_rng(seed)
    max_attempts = 1000
    for attempt in range(max_attempts):
        val_starts = rng.integers(0, max(1, max_start), size=p)
        val_intervals = []
        for vs in val_starts:
            val_intervals.append((vs, min(vs + q, n_intervals)))

        val_intervals = _merge_intervals(val_intervals)
        total_val = sum(end - start for start, end in val_intervals)
        if total_val < n_intervals * 0.05 or total_val > n_intervals * 0.4:
            continue

        train_mask = np.ones(n_intervals, dtype=bool)
        for start, end in val_intervals:
            train_mask[start:end] = False

        train_segments = _get_train_segments(train_mask, T)
        if len(train_segments) == 0:
            continue

        if labels is not None:
            val_labels = np.concatenate(
                [labels[s:e] for s, e in val_intervals])
            train_labels = np.concatenate(
                [labels[s:e] for s, e in train_segments])
            if len(val_labels) == 0 or len(train_labels) == 0:
                continue
            train_balance = np.mean(train_labels[:, 0])
            val_balance = np.mean(val_labels[:, 0])
            if abs(train_balance - val_balance) > 0.10:
                continue
        break
    else:
        q = max(T + 1, n_intervals // 4)
        max_start = n_intervals - q
        val_starts = rng.integers(0, max(1, max_start), size=p)
        val_intervals = []
        for vs in val_starts:
            val_intervals.append((vs, min(vs + q, n_intervals)))
        val_intervals = _merge_intervals(val_intervals)
        train_mask = np.ones(n_intervals, dtype=bool)
        for start, end in val_intervals:
            train_mask[start:end] = False
        train_segments = _get_train_segments(train_mask, T)

    # Trim m bars from the end of each training segment (and from each val
    # block similarly) to prevent label leakage across the train/val boundary.
    # Cm(t) looks m bars ahead, so prediction times in the last m positions
    # of a training segment would have labels that peek into a validation
    # block (or into the gap between segments). Drop those positions.
    train_segments = _trim_segments(train_segments, m)
    val_intervals = _trim_segments(val_intervals, m)

    return val_intervals, train_segments, train_mask, None


def _trim_segments(
    segments: list[tuple[int, int]], m: int
) -> list[tuple[int, int]]:
    """Remove the last m positions from each segment to avoid label leakage.

    A trailing window with prediction time t requires Cm[t] which uses
    vwap[t+m]; if t is within m bars of a segment end, vwap[t+m] lies
    outside the segment and would constitute leakage.
    """
    trimmed = []
    for start, end in segments:
        new_end = end - m
        if new_end > start:
            trimmed.append((start, new_end))
    return trimmed


def _merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    sorted_ints = sorted(intervals, key=lambda x: x[0])
    merged = []
    for start, end in sorted_ints:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(s, e) for s, e in merged]


def _get_train_segments(
    train_mask: np.ndarray, T: int
) -> list[tuple[int, int]]:
    segments = []
    in_segment = False
    start = 0
    for i in range(len(train_mask)):
        if train_mask[i] and not in_segment:
            start = i
            in_segment = True
        elif not train_mask[i] and in_segment:
            if i - start >= T:
                segments.append((start, i))
            in_segment = False
    if in_segment and len(train_mask) - start >= T:
        segments.append((start, len(train_mask)))
    return segments
