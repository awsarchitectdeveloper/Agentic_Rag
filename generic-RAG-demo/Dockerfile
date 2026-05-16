# Use a Python image with uv pre-installed
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

# gcc compiler
RUN apt-get update && \
    apt-get -y install gcc mono-mcs && \
    rm -rf /var/lib/apt/lists/*

# Install the project's dependencies using the lockfile and settings
# Using the --no-install option, we install the dependencies of the project
# but not the project itself. Since the project changes frequently,
# but its dependencies are generally static, this can save time.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev

# Then, add the rest of the project source code and install it
# Installing separately from its dependencies allows optimal layer caching
ADD .python-version config.yaml pyproject.toml uv.lock .
ADD generic_rag ./generic_rag
ADD data ./data

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Install the project as package
RUN uv pip install -e .

# # Place executables in the environment at the front of the path
ENV PATH="/app/.venv/bin:$PATH"

# Reset the entrypoint, don't invoke `uv`
ENTRYPOINT []

EXPOSE 8000

# Run add_sources.py first, then start the chainlit server
CMD python generic_rag/add_sources.py && uv run chainlit run generic_rag/app.py --host 0.0.0.0 --port 8000
