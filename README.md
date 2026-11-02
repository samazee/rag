*This project has been created as part of the 42 curriculum by azgor.*

# RAG against the machine

## Description

This project is a retrieval-augmented generation (RAG) system for the vLLM 0.10.1 codebase. A language model only knows what was in its training data, so the program does not ask Qwen to answer from memory. It indexes the corpus, retrieves the passages that best match a question, and only then asks Qwen/Qwen3-0.6B to write an answer from those passages.

The pipeline has four stages:

1. **Indexing.** Read the files under `data/raw/`, split them into chunks, and store a BM25 index in `data/processed/`.
2. **Retrieval.** Given one question or a JSON dataset, return the top-k source locations. Each location is a file path plus a character span.
3. **Augmenting.** Read those spans back from disk and place them in the model's prompt, inside its token budget.
4. **Generating.** Produce a short answer that follows the retrieved text.

Retrieval quality is measured with recall@k. A reference source counts as found when a retrieved span is in the same file and the two character ranges overlap with an intersection-over-union of at least 0.05.

## Instructions

Requirements: Python 3.11 or later, and [uv](https://docs.astral.sh/uv/). Run every command from the repository root so that corpus paths stay relative (`data/raw/vllm-0.10.1/...`).

```bash
uv sync
```

Or the same step through the Makefile:

```bash
make install
```

The corpus must already be at `data/raw/`. The first answer command downloads `Qwen/Qwen3-0.6B` if it is not cached. Indexing and search do not load the model.

| Make target | What it runs |
| --- | --- |
| `make install` | `uv sync` |
| `make run ARGS="..."` | `uv run python -m src ...` |
| `make debug ARGS="..."` | The same command under `pdb` |
| `make lint` | `flake8 .` and `mypy .` with the project flags |
| `make lint-strict` | `flake8 .` and `mypy . --strict` |
| `make test` | Unit tests (not part of the graded submission) |
| `make clean` | `__pycache__`, `.mypy_cache`, `.pytest_cache` |

Every command is `uv run python -m src <command>`. Input and output paths are arguments. The defaults for the corpus and the index are `data/raw` and `data/processed`.

Empty questions, `k=0`, unknown questions, missing files, and malformed JSON are handled with a message. They do not print a traceback.

## System architecture

`src/` is one Python package, started by `python -m src`.

- `models.py` holds the pydantic models shared by every stage (`MinimalSource`, `RagDataset`, `StudentSearchResults`, `StudentSearchResultsAndAnswer`, and the question and answer types).
- `chunking.py` turns a file into spans. Python and prose use different splitters.
- `index.py` builds and loads the BM25 index and returns ranked `MinimalSource` values.
- `answer.py` reads those spans and calls Qwen/Qwen3-0.6B. The model is imported only when an answer is requested.
- `evaluate.py` computes recall@k for local checks. It does not call the moulinette.
- `cli.py` is the Fire interface. `tqdm` bars cover indexing, dataset search, and dataset answering.

```text
data/raw  -->  chunkers  -->  data/processed/index.pkl
                                     |
                     question -------+--> top-k spans (JSON)
                                     |
                              span text + question
                                     |
                                     v
                              Qwen/Qwen3-0.6B
                                     |
                                     v
                              answer JSON
```

Search results and answers are JSON files validated by the pydantic models before they are written.

## Chunking strategy

A Python file and a Markdown page are not split the same way.

**Python.** The file is parsed with the `ast` module. Each function, class preamble, and module-level statement becomes a block, labelled with its qualified name (`Class.method`). A file that does not parse falls back to one block.

**Markdown and other text.** Markdown is split on ATX headings (`#`, `##`, ...). The header stored for search is the heading path (`LoRA > Load`). Files with no headings (plain text, RST, YAML, and similar) are split on blank lines into paragraphs.

Blocks longer than the working window are cut into overlapping character windows. The CLI flag `--max_chunk_size` is a hard cap (default 2000, and never above 2000, because the moulinette rejects a longer source). When the cap allows it, the window is 600 characters with an overlap of 200. That size is deliberate: a 600-character hit still covers a short reference span (about 40 characters) with IoU above 0.05, and a window that sits inside a long reference (up to about 2000 characters) also clears the bar. Smaller caps are honoured exactly.

The span is an exclusive range, the same convention as a Python slice `text[start:end]`. The path stored on each span is relative to the repository root.

## Retrieval method

Ranking is BM25 (Robertson), with `k1 = 1.5` and `b = 0.75`. It is a lexical method: documents that share rare query terms rank higher, and term frequency saturates so a long file does not win only by repeating a word.

Tokenization keeps three views of each word:

- the whole token, lowercased (`trust_remote_code`, `W4A16Sparse24`)
- snake_case pieces (`trust`, `remote`, `code`)
- camel-case and digit pieces

The indexed text of a chunk is the file path, the structural header (function or heading), and the body. A question can therefore match a path (`triton_flash_attention.py`), a symbol, or the words inside the passage. Identifiers in the question are weighted more heavily. If the file stem itself appears in the question, or several of its distinctive parts do, that file receives an extra bonus. After BM25, the leading candidates are re-scored when the question's identifiers occur verbatim in the span.

Near-duplicate windows of the same passage are removed so the k results are not five copies of one hit. Spans wider than the configured cap, or wider than 2000 characters, are never returned.

## Performance analysis

Measured on the public datasets (100 documentation questions, 99 code questions) with `--max_chunk_size 2000` and `--k 10`. Recall uses the reference files in `data/datasets/AnsweredQuestions/`.

| Stage | Result | Limit |
| --- | --- | --- |
| Index 2157 files (59168 chunks) | 46 s | 5 min |
| Load the index | 1 s | — |
| Search 100 documentation questions | 14.3 s | — |
| Search 99 code questions | 20.8 s | 90 s for 200 questions |

At that pace, 200 questions take well under 90 seconds.

| Dataset | Recall@1 | Recall@3 | Recall@5 | Recall@10 |
| --- | --- | --- | --- | --- |
| Documentation | 0.670 | 0.870 | 0.880 | 0.880 |
| Code | 0.596 | 0.808 | 0.879 | 0.919 |

The required bars are 0.80 recall@5 on documentation and 0.50 recall@5 on code. Both public sets clear them. Documentation questions mostly share words with the passage, so BM25 is strong. Code questions often quote an identifier; keeping the identifier whole, and boosting the file name, is what holds recall up. A few code labels point at a very short or shifted span (one public label is 12 characters and does not contain the identifier the question names). Those misses do not drop code recall under the bar.

Answer quality was checked on the LoRA question. The retrieved spans start at the same offset as the reference (`docs/features/lora.md`, character 4695), and the model answers `/v1/load_lora_adapter`.

## Design decisions

- **BM25 rather than TF-IDF.** Both are allowed. BM25's saturation and length normalisation behave better when many chunks share common words such as "model" or "default", while a rare identifier still dominates.
- **Window of 600, not 2000.** The flag remains a maximum. A 2000-character chunk that merely contains a 40-character reference falls under IoU 0.05 and would be rejected as a hit. Six hundred characters still passes for the short and the long references in the public sets.
- **Headers are searchable but not part of the span.** The path and the function or heading help ranking. The bytes reported to the grader are only the body span.
- **The language model stays out of indexing and search.** Loading Qwen takes longer than the entire retrieval budget. `answer` and `answer_dataset` load it on first use. The default model is `Qwen/Qwen3-0.6B`; `--model_name` can select another one.
- **Greedy decoding, thinking disabled.** Answers should be short and tied to the sources. The prompt tells the model to say it does not know when the sources are silent. Context is dropped from the tail until it fits 3072 tokens.
- **Paths are arguments.** `index`, `search`, `search_dataset`, `answer`, `answer_dataset`, and `evaluate` take their files from the command line. Dataset outputs keep the input file name and go to `--save_directory`, so two datasets can share a name without overwriting each other when the directory is scoped per dataset.

## Challenges faced

The hard part is not generating text. It is returning a span the grader will accept.

Reference spans in the public sets run from a few dozen characters to just under 2000, always in an exact relative path. A chunk that is the right file but the wrong region scores zero. A chunk that contains a tiny region but is much larger than that region also scores zero, because the union swamps the overlap. The window size and the overlap are the compromise that makes both cases pass when the words are actually in the chunk.

Questions and code do not share a vocabulary. "default timeout for RPC" has to find `VLLM_RPC_TIMEOUT`, and "compressed tensors" has to find `compressed_tensors.py`. Splitting identifiers and scoring the path closes most of that gap. Paraphrased documentation questions still fail when a more literal code file outranks the guide; that showed up on a handful of architecture questions whose best lexical hit is the implementation rather than the docs page.

Putting the model in the search process was the other trap. A correct answer that takes two minutes to start cannot meet a 90-second retrieval limit. The index is a pickle of postings, and search never imports `torch`.

## Example usage

Index the corpus:

```bash
uv run python -m src index --max_chunk_size 2000
```

```text
Ingestion complete! Indices saved under data/processed/
```

One question:

```bash
uv run python -m src search \
  "What HTTP endpoint is used to dynamically load a LoRA adapter in vLLM?" \
  --k 3
```

```text
data/raw/vllm-0.10.1/docs/features/lora.md:4695-5295
data/raw/vllm-0.10.1/docs/features/lora.md:5495-6095
data/raw/vllm-0.10.1/docs/features/lora.md:5895-6100
```

Search a dataset. Keep a separate output directory per dataset:

```bash
uv run python -m src search_dataset \
  --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
  --k 10 \
  --save_directory data/output/search_results/UnansweredQuestions
```

```text
Saved student_search_results to data/output/search_results/UnansweredQuestions/dataset_docs_public.json
```

Score that file locally:

```bash
uv run python -m src evaluate \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json
```

Answer one question, then a whole results file:

```bash
uv run python -m src answer \
  "What HTTP endpoint is used to dynamically load a LoRA adapter in vLLM?" \
  --k 2

uv run python -m src answer_dataset \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --save_directory data/output/search_results_and_answer/UnansweredQuestions
```

The single-question answer for the command above is: the endpoint is `/v1/load_lora_adapter`.

The same commands are available as `make run ARGS='search "..." --k 3'`.

## Resources

- Patrick Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* (2020). The paper that defines the retrieve-then-generate setup used here.
- Stephen Robertson and Hugo Zaragoza, *The Probabilistic Relevance Framework: BM25 and Beyond* (2009). The ranking function implemented in `src/index.py`.
- Qwen team, [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B). The required local generator.
- [pydantic](https://docs.pydantic.dev/), [Python Fire](https://github.com/google/python-fire), and [tqdm](https://tqdm.github.io/). Validation, the CLI, and progress bars.
- Python `ast` and PEP 257. Structural chunking for `.py` files, and the docstring convention used in `src/`.

AI was used as a coding assistant on this repository. It was used to compare the implementation with the subject, to write the chunkers, the BM25 index, the CLI, the tests, and this README, and to measure recall on the public datasets and adjust the window size. The ranking choices (BM25, identifier splitting, the 600-character window) and the pydantic models are the parts a reviewer can expect to be explained from the code, not from a generated answer.
