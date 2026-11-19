from build_benchmark_registries import DATASETS, METHODS, TRANSFER_METHODS
from write_benchmark_registry_template import HAIKU_45_PROFILE_ID, template


def test_benchmark_registry_template_has_full_local_and_transfer_matrices() -> None:
    manifest = template()
    assert manifest["status"] == "template_not_results"
    for dataset in DATASETS:
        assert set(manifest["canonical"]["quality"][dataset]) == {
            key for key, _ in METHODS
        }
        assert set(manifest["transfer"]["quality"][dataset]) == {
            key for key, _ in TRANSFER_METHODS
        }
    assert manifest["transfer"]["model_requested"] == HAIKU_45_PROFILE_ID
    assert manifest["canonical"]["endpoint_audit"].endswith(
        "evidencegraph_endpoint_audit.json"
    )
    assert manifest["transfer"]["endpoint_audit"].endswith(
        "haiku45_endpoint_audit.json"
    )
