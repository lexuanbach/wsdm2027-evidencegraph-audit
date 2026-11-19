from sample_paraphrase_validation import sample_aligned_rows


def test_sample_aligned_rows_excludes_prior_ids_and_preserves_alignment() -> None:
    selection = [
        {
            "claim_id": f"c{index}",
            "split": "dev",
            "fever_label": "SUPPORTS",
            "corruption": "duplication",
        }
        for index in range(4)
    ]
    validation = [{"claim_id": f"c{index}", "status": "validated"} for index in range(4)]

    sampled_selection, sampled_validation = sample_aligned_rows(
        selection,
        validation,
        excluded_ids={"c0"},
        limit=2,
        seed=103,
    )

    selected_ids = [record["claim_id"] for record in sampled_selection]
    assert "c0" not in selected_ids
    assert [record["claim_id"] for record in sampled_validation] == selected_ids
