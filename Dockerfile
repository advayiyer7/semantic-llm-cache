FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# uv for fast, reproducible installs
RUN pip install --no-cache-dir uv

# Install deps first for better layer caching
COPY pyproject.toml ./
COPY uv.lock* ./
RUN uv sync --no-dev

COPY app ./app

EXPOSE 8000
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
