# Multi-stage build for Pipeline Guard
FROM python:3.9-slim as builder

# Set working directory
WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.9-slim

# Create non-root user
RUN groupadd -r pipelineguard && useradd -r -g pipelineguard pipelineguard

# Set working directory
WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /root/.local /home/pipelineguard/.local

# Copy application code
COPY . /app

# Set proper permissions
RUN chown -R pipelineguard:pipelineguard /app

# Switch to non-root user
USER pipelineguard

# Add user local bin to PATH
ENV PATH=/home/pipelineguard/.local/bin:$PATH
ENV PYTHONPATH=/app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "from pipeline_guard.core.pipeline_monitor import PipelineMonitor; print('OK')" || exit 1

# Expose port
EXPOSE 8080

# Default command
CMD ["python3", "pipeline_guard_app.py", "start", "--config", "/app/config/production.json"]