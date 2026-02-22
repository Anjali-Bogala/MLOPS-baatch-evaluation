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
