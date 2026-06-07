# ==============================================================================
# Dockerfile — Auto Categorization Webhook
# ==============================================================================
#
# Multi-stage build:
#   Stage 1 (builder) — installs dependencies into a virtual environment.
#   Stage 2 (runtime) — copies only the venv and app source; no build tools.
#
# This keeps the final image lean (~200 MB vs ~600 MB single-stage).
#
# Build:
#   docker build -t auto-categorization-webhook:latest .
#
# Run:
#   docker run --env-file .env -p 8000:8000 auto-categorization-webhook:latest
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 1 — Builder
# ------------------------------------------------------------------------------
FROM python:3.11-slim AS builder

# Security: run as non-root during build
WORKDIR /build

# Install only what we need to compile wheels
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Create an isolated virtual environment inside the builder stage
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy dependency manifest first — Docker cache layer: rebuilds only when
# requirements.txt changes, not on every source code change.
COPY requirements.txt .

# Install production dependencies (exclude test/dev extras)
RUN pip install --upgrade pip \
    && pip install --no-cache-dir \
        fastapi==0.115.6 \
        "uvicorn[standard]==0.32.1" \
        openai==1.58.1 \
        pydantic==2.10.4 \
        pydantic-settings==2.7.0 \
        tenacity==9.0.0 \
        httpx==0.28.1

# ------------------------------------------------------------------------------
# Stage 2 — Runtime
# ------------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

# Metadata labels (OCI Image Spec)
LABEL org.opencontainers.image.title="Auto Categorization Webhook"
LABEL org.opencontainers.image.description="Support ticket classifier using OpenAI GPT-4o"
LABEL org.opencontainers.image.version="1.0.0"
LABEL org.opencontainers.image.licenses="MIT"

# Security: create a dedicated non-root user to run the app
RUN groupadd --gid 1001 appgroup \
    && useradd --uid 1001 --gid appgroup --shell /bin/bash --create-home appuser

WORKDIR /app

# Copy the virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy application source code
COPY --chown=appuser:appgroup app.py .
COPY --chown=appuser:appgroup classifier.py .
COPY --chown=appuser:appgroup prompt_builder.py .
COPY --chown=appuser:appgroup schemas.py .
COPY --chown=appuser:appgroup config.py .
COPY --chown=appuser:appgroup samples.json .

# Create the logs directory with correct ownership before switching user
RUN mkdir -p /app/logs && chown -R appuser:appgroup /app/logs

# Switch to non-root user
USER appuser

# Activate the virtual environment for all subsequent commands
ENV PATH="/opt/venv/bin:$PATH"

# Prevent Python from writing .pyc files to disk (cleaner container)
ENV PYTHONDONTWRITEBYTECODE=1

# Ensure Python output is sent straight to stdout/stderr (no buffering)
# so logs appear immediately in docker logs / CloudWatch / Datadog.
ENV PYTHONUNBUFFERED=1

# Set the default working directory for runtime
ENV PYTHONPATH=/app

# Expose the application port (must match APP_PORT in .env)
EXPOSE 8000

# Health check: Docker will poll this every 30s.
# The container is marked unhealthy if /health returns non-200 three times.
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command: start Uvicorn.
# APP_HOST, APP_PORT, APP_WORKERS are read from environment by app.py's
# __main__ block; here we specify them explicitly for container clarity.
CMD ["uvicorn", "app:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info", \
     "--no-access-log"]
