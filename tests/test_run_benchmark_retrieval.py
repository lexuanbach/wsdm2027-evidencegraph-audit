import pytest

from run_benchmark_retrieval import resolve_candidate_depths, select_case_window


def test_averitec_uses_deeper_coarse_pool_only() -> None:
    assert resolve_candidate_depths(
        "averitec_dev", top_k=20, lexical_candidates=None, rerank_candidates=30
    ) == (1000, 30)
    assert resolve_candidate_depths(
        "hover_dev", top_k=20, lexical_candidates=None, rerank_candidates=30
    ) == (100, 30)


def test_candidate_depths_must_cover_output_depth() -> None:
    with pytest.raises(ValueError, match="positive top-k"):
        resolve_candidate_depths(
            "averitec_dev",
            top_k=20,
            lexical_candidates=10,
            rerank_candidates=30,
        )


def test_select_case_window_supports_disjoint_retrieval() -> None:
    cases = [{"case_id": str(index)} for index in range(8)]
    assert select_case_window(cases, offset=3, limit=2) == cases[3:5]
    with pytest.raises(ValueError):
        select_case_window(cases, offset=-1, limit=2)
