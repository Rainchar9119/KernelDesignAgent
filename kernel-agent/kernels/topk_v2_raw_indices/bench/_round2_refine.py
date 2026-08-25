"""Refinement blocks for round-2 sweep, chosen from the coarse matrix.

Coarse findings driving these:
- Win band in batch is 32..76; batches 1..30 and 77+ show no win (~0.95-1.0).
  -> block C pins the lower onset (30/31/32) and the upper exit (76/77/78).
- Win onset in seq sits between L=98304 (no win) and L=114688 (win).
  -> block A refines 98304..114688.
- Mid-seq anomaly: for B=40 a regression appears at L>=196608 (1.10-1.13); for
  B=48/56/64 there is a no-win gap around L=196608 that recovers by 262144.
  -> block B maps L=163840..262144 for B=40,48,56,64.
Each block is (label, batches, seqs, K).
"""

BLOCKS = [
    ("seq-onset",
     [32, 48, 64, 72, 76],
     [98304, 102400, 106496, 110592, 114688],
     512),
    ("mid-seq-gap",
     [40, 44, 48, 56, 64],
     [163840, 180224, 196608, 212992, 229376, 245760, 262144],
     512),
    ("batch-edges",
     [28, 30, 31, 32, 33, 76, 77, 78, 80],
     [131072, 196608, 262144],
     512),
]
