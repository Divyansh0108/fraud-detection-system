# 1. Base Image
FROM python:3.12-slim-bookworm

# 2. Env Vars
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 3. System Deps
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# 4. Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# 5. Create Non-Root User FIRST
RUN useradd -m appuser

# 6. Set Workdir & Ownership
WORKDIR /app
RUN chown appuser:appuser /app

# 7. Switch to User (All subsequent commands run as 'appuser')
USER appuser

# 8. Copy Config Files (With correct ownership!)
COPY --chown=appuser:appuser pyproject.toml uv.lock ./

# 9. Install Dependencies (As appuser, so .venv is owned by appuser)
RUN uv sync --frozen --no-dev

# 10. Copy Code & Artifacts
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser models/ ./models/

# 11. Expose Port
EXPOSE 8000

# 12. Run Uvicorn DIRECTLY (Faster, bypassing 'uv run')
CMD ["/app/.venv/bin/uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]