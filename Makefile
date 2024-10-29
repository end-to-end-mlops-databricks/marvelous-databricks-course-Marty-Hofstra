init:
	uv venv -p 3.11.9 venv && \
	source venv/bin/activate && \
	uv pip install -r pyproject.toml --all-extras && \
	uv lock

get_package_name:
	@python -c "import toml; print(toml.load('pyproject.toml')['project']['name'])"

get_package_version:
	@python -c "import toml; print(toml.load('pyproject.toml')['project']['version'])"

build_and_store_whl:
	uv build
	@name=$$(make get_package_name); \
	version=$$(make get_package_version); \
	databricks fs cp dist/$$name-$$version-py3-none-any.whl ${dbfs_path}
