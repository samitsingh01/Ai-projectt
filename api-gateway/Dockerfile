FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Create non-root user, install dependencies, and clean up in one layer
RUN groupadd -g 1001 appuser && \
    useradd -u 1001 -g appuser -s /bin/bash -m appuser && \
    apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    pip install --no-cache-dir \
        fastapi==0.109.0 \
        uvicorn==0.27.0 \
        httpx==0.26.0 \
        python-multipart==0.0.6 \
        pydantic==2.5.3 \
        mysql-connector-python==8.2.0 && \
    apt-get purge -y gcc && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# Set working directory and copy application
WORKDIR /app
COPY --chown=appuser:appuser main.py .

USER appuser
EXPOSE 8000

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
