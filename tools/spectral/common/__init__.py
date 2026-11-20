"""Shared helpers of the spectral-measurement suite.

Modules:
  model_io        repository root, model plug-in import, checkpoint loading report, seeding
  spectral_core   graph-spectral primitives (band edges, Chebyshev-Jackson filters, exact DCT bins, HFER metrics)
  graph_spectral  plane-level graph spectra (full-plane DCT and teacher-support sub-graph operators)
  tpv_metrics     TPV-plane spectral metrics (tight-frame band split, shares, CKA, regional statistics)
  logits_core     band-limited signals on the voxel support (3D) and plane support (2D), ridge recovery, AUROC
  plane_cells     plane-cell semantic categories and per-plane point counts

All numerics are the ones used to produce the paper; see the README of each stage.
"""
