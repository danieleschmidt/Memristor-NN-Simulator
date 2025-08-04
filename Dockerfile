# Multi-stage build for production deployment
FROM python:3.9-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r memristor && useradd -r -g memristor memristor

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY pyproject.toml README.md ./

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install .

# Development stage
FROM base as development

# Install development dependencies
RUN pip install -e ".[dev,rtl]"

# Copy source code
COPY . .

# Change ownership to non-root user
RUN chown -R memristor:memristor /app

USER memristor

CMD ["python", "-m", "memristor_nn.examples.basic_usage"]

# Production stage
FROM base as production

# Copy only necessary files
COPY memristor_nn/ ./memristor_nn/
COPY examples/ ./examples/

# Create cache directory
RUN mkdir -p /app/cache && chown -R memristor:memristor /app

USER memristor

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import memristor_nn; print('OK')" || exit 1

CMD ["python", "-m", "memristor_nn.examples.basic_usage"]

# Benchmark stage for performance testing
FROM production as benchmark

USER root
RUN pip install -e ".[dev]"

USER memristor

CMD ["python", "-c", "from memristor_nn.optimization import BenchmarkSuite; suite = BenchmarkSuite(); print('Benchmark ready')"]