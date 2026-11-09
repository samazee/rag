.PHONY: install run debug clean lint lint-strict

ARGS ?=

install:
	uv sync

run:
	uv run python -m src $(ARGS)

debug:
	uv run python -m pdb -m src $(ARGS)

clean:
	find . -path './src' -prune -o -type d -name '__pycache__' -exec rm -rf {} +
	rm -rf .mypy_cache .flake8_cache

lint:
	uv run flake8 src
	uv run mypy src --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 src
	uv run mypy src --strict

test:
	#rm data/processed/index.json
	#uv run python -m src index --max_chunk_size 2000
	#uv run python -m src search "What backend allows many decoder language models to be automatically loaded in vLLM without manual implementation?" --k 5
	#uv run python -m src answer "What backend allows many decoder language models to be automatically loaded in vLLM without manual implementation?" --k 5
	#uv run python -m src search_dataset --dataset_path data/datasets/UnansweredQuestions/dataset_code_public.json --save_directory data/output/UnansweredQuestions
	#uv run python -m src answer_dataset --student_search_results_path data/output/UnansweredQuestions/dataset_code_public.json --save_directory data/output/AnsweredQuestions
	uv run python -m src evaluate --student_search_results_path data/output/AnsweredQuestions/dataset_code_public.json --dataset_path data/datasets/AnsweredQuestions/dataset_code_public.json
