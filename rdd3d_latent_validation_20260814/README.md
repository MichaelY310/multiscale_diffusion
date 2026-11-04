# RDD-3D latent-to-mesh validation

This directory contains a positive-control decode for asset
`060817eb9a66b0a90ff517e50ef3aa7c3179bf2bcdc3407ec6d2cebb64f74724`.

- `gt_slat.npz`: the official TRELLIS.2 SC-VAE encoder output (`coords` plus
  32-D shape SLat features) at latent resolution 32.
- `gt_slat_preview.png`: the conditioning render beside four normal renders
  of the decoded mesh.
- `gt_slat_decoded_light.glb`: a 200K-face viewing copy of the decoded mesh.
- `gt_slat_decoded.glb`: the full 960,616-face decoder output. This file is
  valid only when its size is exactly 17,286,876 bytes; use the light copy if
  a prior interrupted transfer left it incomplete.

The latent was decoded with the same unmodified decoder and export path used
for the 100K RDD predictions:

```text
microsoft/TRELLIS.2-4B/ckpts/shape_dec_next_dc_f16c32_fp16
latent resolution 32 -> SC-VAE 16x decode -> O-Voxel resolution 512 -> mesh
```

The positive control produces a coherent person mesh (479,890 vertices,
960,616 faces, and only 2 connected components). The RDD prediction for the
same condition has 1,458 connected components before lightweight GLB
simplification. This rules out the ordinary
SLat-to-SC-VAE-to-mesh/export path as the primary source of fragmentation. It
does not by itself rule out an RDD/decoder interface distribution mismatch;
predicted support and predicted features still need separate controls.
