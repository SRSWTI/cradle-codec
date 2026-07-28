import unittest

import numpy as np

from cradle_codec.layout import (
    HeadDimTiling,
    KVShape,
    candidate_name_for_tiling,
    estimate_raw_payload_bytes,
    layout_candidates_for_shape,
    layout_from_name,
    profile_layout_candidates,
    select_layout_candidate,
    validate_candidate_name,
)


class LayoutCandidateTests(unittest.TestCase):
    def test_candidate_name_roundtrip_and_suffix_validation(self) -> None:
        tiling = validate_candidate_name("h1x8_d32x4_32x32", num_kv_heads=8, head_dim=128)

        self.assertEqual(tiling, HeadDimTiling(head_rows=1, head_cols=8, dim_rows=32, dim_cols=4))
        self.assertEqual(candidate_name_for_tiling(tiling), "h1x8_d32x4_32x32")
        self.assertEqual(candidate_name_for_tiling(tiling, paper_prefix=True), "paper_h1x8_d32x4_32x32")
        with self.assertRaisesRegex(ValueError, "geometry suffix mismatch"):
            validate_candidate_name("h1x8_d32x4_16x64", num_kv_heads=8, head_dim=128)
        with self.assertRaisesRegex(ValueError, "head tiling"):
            validate_candidate_name("h3x3_d32x4_96x12", num_kv_heads=8, head_dim=128)

    def test_layout_candidates_for_shape_are_deterministic_and_valid(self) -> None:
        shape = KVShape(num_sides=2, num_layers=3, num_tokens=5, num_kv_heads=4, head_dim=8)

        layouts = layout_candidates_for_shape(shape, layers_per_frame=2)
        names = tuple(candidate_name_for_tiling(layout.tiling) for layout in layouts)

        self.assertEqual(len(layouts), 12)
        self.assertEqual(names[:4], ("h1x4_d1x8_1x32", "h1x4_d2x4_2x16", "h1x4_d4x2_4x8", "h1x4_d8x1_8x4"))
        self.assertEqual(names[-1], "h4x1_d8x1_32x1")
        self.assertTrue(all(layout.num_layers == shape.num_layers for layout in layouts))
        self.assertTrue(all(layout.layers_per_frame == 2 for layout in layouts))

    def test_layout_from_name_accepts_prefixed_canonical_name(self) -> None:
        layout = layout_from_name("paper_h2x4_d16x8_32x32", num_layers=36, num_kv_heads=8, head_dim=128)

        self.assertEqual(layout.name, "paper_h2x4_d16x8_32x32")

    def test_estimate_raw_payload_bytes_matches_packed_shape(self) -> None:
        shape = KVShape(num_sides=2, num_layers=2, num_tokens=3, num_kv_heads=2, head_dim=4)
        layout = layout_from_name("h1x2_d2x2_2x4", num_layers=2, num_kv_heads=2, head_dim=4, layers_per_frame=1)

        self.assertEqual(estimate_raw_payload_bytes(shape, layout, dtype=np.uint8), 288)

    def test_profiler_scores_and_selects_with_injected_encoder(self) -> None:
        kv = np.arange(2 * 2 * 2 * 2 * 4, dtype=np.uint8).reshape(2, 2, 2, 2, 4)
        candidates = ("h1x2_d1x4_1x8", "h2x1_d4x1_8x1")

        def encode_by_height(frames: np.ndarray, geometry) -> int:  # type: ignore[no-untyped-def]
            del frames
            return geometry.logical_height

        profiles = profile_layout_candidates(kv, candidates, layers_per_frame=1, encode=encode_by_height)
        selected = select_layout_candidate(kv, candidates, layers_per_frame=1, encode=encode_by_height)

        self.assertEqual(tuple(profile.name for profile in profiles), ("h1x2_d1x4_1x8", "h2x1_d4x1_8x1"))
        self.assertEqual(profiles[0].encoded_payload_bytes, 4)
        self.assertEqual(profiles[0].raw_payload_bytes, 192)
        self.assertEqual(profiles[0].part_count, 4)
        self.assertEqual(selected.name, "h1x2_d1x4_1x8")


if __name__ == "__main__":
    unittest.main()
