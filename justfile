# Run a specific python script
run script:
    uv run python {{script}}

# Run a shell script
sh script:
    uv run bash {{script}}

# Run Jupyter Lab
notebook:
    uv run jupyter lab

# Run lint and format
lint:
    uv run ruff check --fix
    uv run ruff format --preview

# Sync dependencies
sync:
    uv sync --all-groups

# Clean all caches
clean:
    find . -type d \( -name "__pycache__" -o -name ".ruff_cache" \) -exec rm -rf {} +
