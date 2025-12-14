# EdgeAgent SubAgent Docker Image
# Runs a location-specific SubAgent server (EDGE or CLOUD)

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy edgeagent package and config
COPY edgeagent/ /app/edgeagent/
COPY config/ /app/config/

# Set Python path
ENV PYTHONPATH=/app

# Default port
ENV PORT=8080

# LOCATION must be set at runtime (EDGE or CLOUD)
# CONFIG_PATH can be overridden (default: /app/config/tools_docker.yaml)
ENV CONFIG_PATH=/app/config/tools_docker.yaml

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

EXPOSE 8080

# Run SubAgent server
# LOCATION environment variable must be provided
CMD python -m edgeagent.subagent \
    --location ${LOCATION} \
    --port ${PORT} \
    --config ${CONFIG_PATH}
