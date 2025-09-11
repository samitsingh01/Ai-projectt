# intelligent-qa-service/Dockerfile - Improved with proper service integration
FROM python:3.11-slim AS builder

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Runtime
FROM python:3.11-slim

# Install runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Copy Python dependencies from builder
COPY --from=builder /root/.local /home/appuser/.local

# Create necessary directories
RUN mkdir -p /app/logs /app/services /app/temp && \
    chown -R appuser:appuser /app

# Copy application code
COPY --chown=appuser:appuser main.py .
COPY --chown=appuser:appuser services/ ./services/

# Set Python path and environment variables
ENV PATH=/home/appuser/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 6000

# Health check with better timeout
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:6000/health', timeout=5)" || exit 1

# Use the ORIGINAL Dockerfile, just change the CMD:
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "6000"]
