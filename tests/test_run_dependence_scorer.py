from __future__ import annotations

import json

import numpy as np
import pytest

from run_dependence_scorer import write_predictions_atomic


def test_dependence_predictions_stream_atomically(tmp_path) -> None:
    output = tmp_path / "predictions.jsonl"
    metadata = [
        {
            "claim_id": "c1",
            "left_id": "a",
            "right_id": "b",
            "left_family": "independent",
            "right_family": "mirror",
        },
        {
            "claim_id": "c2",
            "left_id": "c",
            "right_id": "d",
            "left_family": "citation",
            "right_family": "paraphrase",
        },
    ]
    written = write_predictions_atomic(
        output,
        (103, 104, 105),
        [
            (
                "test",
                np.asarray([1, 0]),
                metadata,
                [
                    np.asarray([0.9, 0.1]),
                    np.asarray([0.8, 0.2]),
                    np.asarray([0.7, 0.3]),
                ],
            )
        ],
    )

    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert written == 2
    assert rows[0]["probabilities"] == {"103": 0.9, "104": 0.8, "105": 0.7}
    assert not (tmp_path / ".predictions.jsonl.partial").exists()


def test_dependence_prediction_writer_rejects_misaligned_rows(tmp_path) -> None:
    with pytest.raises(ValueError, match="do not align"):
        write_predictions_atomic(
            tmp_path / "predictions.jsonl",
            (103, 104, 105),
            [("test", np.asarray([1]), [], [np.asarray([0.5])] * 3)],
        )
