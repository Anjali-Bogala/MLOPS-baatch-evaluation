# MLOps Engineering Internship: Technical Assessment
MetaStackerBandit Project

This repository contains a complete MLOps pipeline implementation demonstrating core principles of reproducibility, containerized deployment, comprehensive logging, and structured metrics output. The pipeline processes cryptocurrency OHLCV data to generate trading signals based on rolling mean analysis.

 Overview

The pipeline implements the following functionality:

- **Deterministic Execution**: Configuration-driven with reproducible random seeds
- **Rolling Mean Analysis**: Calculates moving averages on closing prices
- **Signal Generation**: Binary signals (1 = price above mean, 0 = price at or below mean)
- **Structured Metrics**: Machine-readable JSON output with performance characteristics
- **Comprehensive Logging**: Complete execution trace with timestamps and status messages

Prerequisites

- Python 3.9 or higher
- Docker (for containerized deployment)
- Git (for cloning the repository)

Setup Instructions

Local Development Setup

```bash
# Clone the repository
git clone <repository-url>
cd mlops-assessment

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
# MLOps Signal Detection Pipeline

**Production CLI pipeline** processing CSV data → rolling window analysis → JSON metrics (13ms latency)

##  PRODUCTION RESULTS
signal_rate=0.5 | 2 rows processed | window=5 | seed=42 | 13ms latency

##  LIVE EXECUTION LOG
2026-02-23 17:19:54 - INFO - Config loaded: seed=42, window=5, version=v1
2026-02-23 17:19:54 - INFO - Data loaded: 2 rows
2026-02-23 17:19:54 - INFO - Metrics: signal_rate=0.5000, rows_processed=2
2026-02-23 17:19:54 - INFO - Job completed successfully in 13ms


##  RUN LOCALLY
```bash
pip install -r requirements.txt
python run.py --input data.csv --config config.yaml --output metrics.json --log-file run.log
