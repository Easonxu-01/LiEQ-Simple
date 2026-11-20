"""Frame selection of the spectral measurements.

Paper rule (validation split of N frames, dataset indices):
  lin    = round(linspace(0, N-1, 200))                      the 200-frame grid (numpy rounding)
  dev    = {0, N//3, 2N//3}                                 development frames
  eval   = lin minus dev                                    the frames that enter the statistics (199 in the paper)
  all    = lin union dev                                    the frames that are dumped and summarised (202)
  review = lin[round(linspace(1, 199, n))], n = 10          evenly spaced frames of the grid
  raw    = dev union review                                 frames whose full tensors are also written (13)
Development frames are dumped and summarised like every other frame but are excluded from the statistics;
frame 0 is both on the grid and a development frame, hence 199 evaluated frames.
Raw frames are summarised in 'full' mode (all metrics); all other frames in 'eval' mode.
"""
import json
import os

import numpy as np

N_GRID = 200
N_REVIEW = 10


def lin_frames(N, n=N_GRID):
    return [int(x) for x in np.round(np.linspace(0, N - 1, n)).astype(int)]


def dev_frames(N):
    return [0, N // 3, (2 * N) // 3]


def review_frames(N, n=N_REVIEW):
    lin = lin_frames(N)
    if n <= 0:
        return []
    idx = np.round(np.linspace(1, len(lin) - 1, n)).astype(int)
    return [lin[i] for i in idx]


def paper_frames(N):
    lin, dev = lin_frames(N), dev_frames(N)
    ev = [f for f in lin if f not in dev]
    review = review_frames(N)
    return dict(N=int(N), lin=lin, dev=dev, eval=ev, n_eval=len(ev), all=sorted(set(lin) | set(dev)),
                review=review, raw=sorted(set(dev) | set(review)))


def parse_frame_list(spec):
    """A file (JSON list, or integers separated by commas / white space) or a comma-separated list."""
    if os.path.isfile(spec):
        text = open(spec).read().strip()
        if text.startswith('['):
            return [int(x) for x in json.loads(text)]
        return [int(x) for x in text.replace(',', ' ').split()]
    return [int(x) for x in spec.split(',') if x.strip()]


def raw_frames(spec, N):
    """'paper' = dev U review(10); 'none' = no raw frames; an integer n = dev U review(n) (0 = none);
    otherwise a frame list or file (see parse_frame_list)."""
    if spec in (None, '', 'paper'):
        return paper_frames(N)['raw']
    if spec == 'none':
        return []
    if spec.isdigit():
        n = int(spec)
        return [] if n == 0 else sorted(set(dev_frames(N)) | set(review_frames(N, n)))
    return sorted(set(parse_frame_list(spec)))


def shard(frames, spec):
    """'I/N': the I-th of N interleaved shards of the sorted frame list (per-frame outputs do not depend on sharding)."""
    i, n = (int(x) for x in spec.split('/'))
    if not (n >= 1 and 0 <= i < n):
        raise SystemExit(f'--shard must be I/N with 0 <= I < N: {spec}')
    return sorted(frames)[i::n]
