# Third-party notices

The MIT licence in `LICENSE` covers the code and documentation written by the
authors. It does not relicense benchmark data, model weights, publisher text,
or dependencies.

This release contains aggregate numerical results, frozen numeric predictions,
and deidentified numeric audit labels. It contains no third-party passage text.
Claim identifiers that appear in stored outputs refer to the public benchmarks
below and keep those benchmarks' terms.

| Resource | Used for | Where to obtain it |
|---|---|---|
| FEVER and its 2017 Wikipedia pages | construction and FEVER evaluation | https://fever.ai |
| HoVer | HoVer evaluation | https://hover-nlp.github.io |
| AVeriTeC and its knowledge store | AVeriTeC evaluation | https://fever.ai/dataset/averitec.html |
| Wikidata dump of 2017-02-20 | provenance terms for construction | https://www.wikidata.org |
| Hugging Face models named in `README.md` | NLI, retrieval and reranking | https://huggingface.co |
| Qwen2.5 models via Ollama | paraphrase generation, duplication intervention | https://ollama.com |
| Claude Haiku 4.5 on Amazon Bedrock | end-to-end answering | https://aws.amazon.com/bedrock |

Obtain these resources from their owners under their own terms. The presence of
code that reads a resource is not permission to redistribute it. Python
dependencies are installed from their upstream distributions and keep their own
licences; none is vendored here.
