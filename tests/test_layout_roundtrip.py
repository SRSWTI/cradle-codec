import unittest

import numpy as np

from cradle_codec.layout import (
    HeadDimTiling,
    KVCodecLayout,
    layout_from_name,
    pack_kv_to_frame_batches,
    qwen3_8b_layout_candidates,
    unpack_frame_batches_to_kv,
    validate_layout,
)


def synthetic_kv(num_layers: int, num_tokens: int, num_kv_heads: int, head_dim: int) -> np.ndarray:
    """Unique-axis values make swapped side/layer/token/head/dim immediately visible."""

    side = np.arange(2, dtype=np.uint64)[:, None, None, None, None] * 10_000_000_000
    layer = np.arange(num_layers, dtype=np.uint64)[None, :, None, None, None] * 100_000_000
    token = np.arange(num_tokens, dtype=np.uint64)[None, None, :, None, None] * 1_000_000
    head = np.arange(num_kv_heads, dtype=np.uint64)[None, None, None, :, None] * 10_000
    dim = np.arange(head_dim, dtype=np.uint64)[None, None, None, None, :]
    return side + layer + token + head + dim


class LayoutRoundtripTests(unittest.TestCase):
    def test_tiny_layout_roundtrip_preserves_values(self) -> None:
        layout = KVCodecLayout(
            num_layers=2,
            num_kv_heads=2,
            head_dim=4,
            layers_per_frame=3,
            tiling=HeadDimTiling(head_rows=1, head_cols=2, dim_rows=2, dim_cols=2),
        )
        kv = synthetic_kv(num_layers=2, num_tokens=3, num_kv_heads=2, head_dim=4)

        batches = pack_kv_to_frame_batches(kv, layout)
        restored = unpack_frame_batches_to_kv(batches, layout, num_tokens=3)

        self.assertEqual(len(batches), 2)  # K and V, one partial layer group each.
        self.assertTrue(np.array_equal(restored, kv))

    def test_qwen3_like_layout_roundtrip_preserves_values(self) -> None:
        layout = layout_from_name(
            "h1x8_d32x4_32x32",
            num_layers=36,
            num_kv_heads=8,
            head_dim=128,
            layers_per_frame=3,
        )
        kv = synthetic_kv(num_layers=36, num_tokens=5, num_kv_heads=8, head_dim=128)

        batches = pack_kv_to_frame_batches(kv, layout)
        restored = unpack_frame_batches_to_kv(batches, layout, num_tokens=5)

        self.assertEqual(len(batches), 24)  # 2 sides * ceil(36 / 3).
        self.assertEqual(batches[0].frames.shape, (5, 32, 32, 3))
        self.assertTrue(np.array_equal(restored, kv))

    def test_partial_layer_group_roundtrip_preserves_values(self) -> None:
        layout = KVCodecLayout(
            num_layers=4,
            num_kv_heads=2,
            head_dim=4,
            layers_per_frame=3,
            tiling=HeadDimTiling(head_rows=2, head_cols=1, dim_rows=1, dim_cols=4),
        )
        kv = synthetic_kv(num_layers=4, num_tokens=2, num_kv_heads=2, head_dim=4)

        batches = pack_kv_to_frame_batches(kv, layout)
        restored = unpack_frame_batches_to_kv(batches, layout, num_tokens=2)

        self.assertEqual(len(batches), 4)  # 2 sides * 2 layer groups.
        final_k = batches[1]
        self.assertEqual(final_k.metadata.layer_group.layer_indices, (3,))
        self.assertTrue(np.all(final_k.frames[..., 1] == 0))
        self.assertTrue(np.all(final_k.frames[..., 2] == 0))
        self.assertTrue(np.array_equal(restored, kv))

    def test_invalid_head_tiling_rejected(self) -> None:
        layout = KVCodecLayout(
            num_layers=36,
            num_kv_heads=8,
            head_dim=128,
            layers_per_frame=3,
            tiling=HeadDimTiling(head_rows=3, head_cols=3, dim_rows=32, dim_cols=4),
        )
        with self.assertRaisesRegex(ValueError, "head tiling"):
            validate_layout(layout)

    def test_invalid_dim_tiling_rejected(self) -> None:
        layout = KVCodecLayout(
            num_layers=36,
            num_kv_heads=8,
            head_dim=128,
            layers_per_frame=3,
            tiling=HeadDimTiling(head_rows=1, head_cols=8, dim_rows=7, dim_cols=18),
        )
        with self.assertRaisesRegex(ValueError, "dim tiling"):
            validate_layout(layout)

    def test_layer_groups_cover_all_layers_with_partial_final_group(self) -> None:
        layout = KVCodecLayout(
            num_layers=5,
            num_kv_heads=1,
            head_dim=4,
            layers_per_frame=3,
            tiling=HeadDimTiling(head_rows=1, head_cols=1, dim_rows=2, dim_cols=2),
        )

        groups = layout.layer_groups()

        self.assertEqual([group.layer_indices for group in groups], [(0, 1, 2), (3, 4)])
        self.assertEqual(tuple(layer for group in groups for layer in group.layer_indices), (0, 1, 2, 3, 4))

    def test_token_axis_becomes_frame_axis(self) -> None:
        layout = KVCodecLayout(
            num_layers=3,
            num_kv_heads=1,
            head_dim=4,
            layers_per_frame=3,
            tiling=HeadDimTiling(head_rows=1, head_cols=1, dim_rows=2, dim_cols=2),
        )
        kv = synthetic_kv(num_layers=3, num_tokens=4, num_kv_heads=1, head_dim=4)

        batch = pack_kv_to_frame_batches(kv, layout)[0]

        self.assertEqual(batch.metadata.side_name, "k")
        self.assertEqual(batch.frames.shape[0], 4)
        # Same layer/head/dim location across consecutive frames differs only by token value.
        self.assertEqual(batch.frames[0, 0, 0, 0], kv[0, 0, 0, 0, 0])
        self.assertEqual(batch.frames[1, 0, 0, 0], kv[0, 0, 1, 0, 0])
        self.assertEqual(batch.frames[2, 0, 0, 0], kv[0, 0, 2, 0, 0])
        self.assertEqual(batch.frames[3, 0, 0, 0], kv[0, 0, 3, 0, 0])

    def test_multiple_required_tilings_roundtrip_preserve_values(self) -> None:
        names = (
            "h1x8_d32x4_32x32",
            "h1x8_d8x16_8x128",
            "h2x4_d16x8_32x32",
        )
        kv = synthetic_kv(num_layers=6, num_tokens=3, num_kv_heads=8, head_dim=128)

        for name in names:
            with self.subTest(name=name):
                layout = layout_from_name(name, num_layers=6, num_kv_heads=8, head_dim=128)
                restored = unpack_frame_batches_to_kv(pack_kv_to_frame_batches(kv, layout), layout, num_tokens=3)
                self.assertTrue(np.array_equal(restored, kv))

    def test_qwen3_candidates_are_valid(self) -> None:
        candidates = qwen3_8b_layout_candidates()

        self.assertEqual([candidate.tiling.name for candidate in candidates], [
            "h1x8_d32x4",
            "h1x8_d16x8",
            "h1x8_d8x16",
            "h2x4_d16x8",
            "h4x2_d8x16",
        ])
        self.assertTrue(all(candidate.num_layers == 36 for candidate in candidates))


if __name__ == "__main__":
    unittest.main()
