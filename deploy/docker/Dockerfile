# ==============================================================================
# Stoic Citadel - Production Container with Multi-Stage Build
# ==============================================================================
# Production container for running Freqtrade with Stoic Citadel enhancements
# Multi-stage build for smaller image size and reduced attack surface
# ==============================================================================

# Stage 1: Builder - install dependencies and build wheels
FROM python:3.12-slim AS builder

# Install system dependencies for building Python packages
RUN apt-get update && apt-get install -y \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install TA-Lib from source (required for technical analysis)
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib/ && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    cd .. && \
    rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

# Set working directory
WORKDIR /app

# Copy dependency specification
COPY pyproject.toml .

# Create virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install build dependencies and the package
RUN pip install --upgrade pip && \
    pip install wheel && \
    pip install freqtrade>=2024.11 && \
    pip install TA-Lib>=0.4.28 && \
    pip install -e .

# Stage 2: Runner - minimal production image
FROM python:3.12-slim AS runner

# Install runtime dependencies only (no build tools)
RUN apt-get update && apt-get install -y \
    wget \
    # Required for TA-Lib runtime
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy TA-Lib libraries from builder
COPY --from=builder /usr/lib/libta_lib.* /usr/lib/
COPY --from=builder /usr/include/ta-lib/ /usr/include/ta-lib/

# Copy Python virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /freqtrade

# Copy application source code
# We copy to /freqtrade/src so that 'import src.xxx' works when PYTHONPATH=/freqtrade
COPY src/ /freqtrade/src/
COPY deploy/scripts/ /freqtrade/scripts/
COPY config/ /freqtrade/config/
COPY manage.py /freqtrade/

# Create necessary directories
RUN mkdir -p /freqtrade/user_data/{logs,data,strategies,config,backtest_results}

# Copy production config
COPY user_data/config/config_production.json /freqtrade/user_data/config/config_production.json

RUN ls -la /freqtrade/user_data/config

# Set environment variables
# Add /freqtrade to PYTHONPATH so we can import src.xxx and run manage.py
ENV PYTHONPATH=/freqtrade:${PYTHONPATH}
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=utf-8
ENV FREQTRADE__USER_DATA_PATH=/freqtrade/user_data

# Create non-root user for security
RUN groupadd -r freqtrade && useradd -r -g freqtrade freqtrade && \
    chown -R freqtrade:freqtrade /freqtrade
USER freqtrade

# Expose Freqtrade web server port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8080/api/v1/ping', timeout=2)" || exit 1

# Default command: Use manage.py trade
CMD ["python", "manage.py", "trade"]