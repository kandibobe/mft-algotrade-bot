# ==============================================================================
# Stoic Citadel - Freqtrade Production Container
# ==============================================================================
# Production container for running Freqtrade with Stoic Citadel enhancements
# ==============================================================================

FROM python:3.11-slim

# Set working directory
WORKDIR /freqtrade

# Install system dependencies for TA-Lib and other required libraries
RUN apt-get update && apt-get install -y \
    wget \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install TA-Lib
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib/ && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    cd .. && \
    rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

# Copy requirements files
COPY requirements.txt /tmp/requirements.txt
COPY requirements-dev.txt /tmp/requirements-dev.txt

# Install freqtrade first (it includes many dependencies)
RUN pip install --no-cache-dir freqtrade>=2024.11

# Install additional dependencies from requirements.txt
# Filter out pandas-ta if it causes issues, install it separately if needed
RUN pip install --no-cache-dir \
    pandas>=2.0.0 \
    numpy>=1.24.0 \
    TA-Lib>=0.4.28 \
    scikit-learn>=1.3.0 \
    sqlalchemy>=2.0.0 \
    psycopg2-binary>=2.9.0 \
    redis>=5.0.0 \
    ccxt>=4.0.0 \
    aiohttp>=3.9.0 \
    structlog>=24.0.0 \
    prometheus-client>=0.19.0 \
    numba>=0.58.0 \
    matplotlib>=3.7.0 \
    seaborn>=0.13.0

# pandas-ta is optional, skip if not compatible
# RUN pip install --no-cache-dir pandas-ta==0.3.14b0

# Copy project source code
COPY src/ /freqtrade/user_data/src/
COPY scripts/ /freqtrade/scripts/
COPY config/ /freqtrade/config/

# Create necessary directories
RUN mkdir -p /freqtrade/user_data/{logs,data,strategies,config,backtest_results}

# Set environment variables
ENV PYTHONPATH=/freqtrade/user_data/src
ENV PYTHONUNBUFFERED=1
ENV FREQTRADE__USER_DATA_PATH=/freqtrade/user_data

# Expose Freqtrade web server port
EXPOSE 8080

# Default command (can be overridden by docker-compose)
CMD ["freqtrade", "--version"]
