# Prerequisites:
#   - uv (Python package manager)
#   - databricks CLI

.DEFAULT_GOAL := help

.PHONY: help
help:
	@echo "Available targets:"
	@echo "  init                 - Synchronize dependencies using uv"
	@echo "  build_and_store_whl - Build package and store in DBFS"
	@echo "  pre_commit_all_files - Run pre-commit checks on all files"

init:
	uv sync $(if $(extra),--extra=$(extra))

get_package_name:
	@python -c "import toml; print(toml.load('pyproject.toml')['project']['name'])"

get_package_version:
	@python -c "import toml; print(toml.load('pyproject.toml')['project']['version'])"

build_and_store_whl:
	uv build --python-preference managed
	@name=$$(make get_package_name); \
	version=$$(make get_package_version); \
	databricks fs cp dist/$$name-$$version-py3-none-any.whl ${dbfs_path} --overwrite

pre_commit_all_files:
	uv run pre-commit run --all-files
