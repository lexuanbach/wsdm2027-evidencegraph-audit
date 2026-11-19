from __future__ import annotations

import numpy as np
import pytest

from evidencegraph.bootstrap import multinomial_weight_batches


def test_multinomial_weight_batches_match_one_shot_seeded_draws() -> None:
    n_items = 17
    n_resamples = 257
    expected = np.random.default_rng(103).multinomial(
        n_items,
        np.full(n_items, 1.0 / n_items),
        size=n_resamples,
    )
    observed = np.concatenate(
        list(
            multinomial_weight_batches(
                np.random.default_rng(103),
                n_items,
                n_resamples,
                batch_size=31,
            )
        ),
        axis=0,
    )

    np.testing.assert_array_equal(observed, expected)


def test_multinomial_weight_batches_reject_nonpositive_dimensions() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        list(multinomial_weight_batches(np.random.default_rng(103), 0, 100))
