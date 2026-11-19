from evidencegraph.nli import (
    DEBERTA_LARGE_MODEL,
    DEBERTA_LARGE_REVISION,
    MINILM_NLI_MODEL,
    MINILM_NLI_REVISION,
)


def test_frozen_nli_model_identities_match_their_pinned_revisions() -> None:
    assert (
        DEBERTA_LARGE_MODEL
        == "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
    )
    assert DEBERTA_LARGE_REVISION == "b3546ea6b0346eb6f8d5d68b13c7dc6d0376b3d7"
    assert MINILM_NLI_MODEL == "cross-encoder/nli-MiniLM2-L6-H768"
    assert MINILM_NLI_REVISION == "b95119ce93d3e065de6214e38cd4a97b0f2f2c6d"
