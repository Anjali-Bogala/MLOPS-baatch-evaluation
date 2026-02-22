#!/usr/bin/env python3
"""
MLOps Pipeline: Rolling Mean Signal Generator

This script implements a deterministic MLOps pipeline for processing cryptocurrency
OHLCV data and generating trading signals based on rolling mean analysis.

Key Features:
- Configuration-driven execution for reproducibility
- Comprehensive logging with structured output
- Machine-readable metrics in JSON format
- Robust error handling with graceful degradation

Usage:
    python run.py --input data.csv --config config.yaml --output metrics.json --log-file run.log

Version: v1
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import yaml


def setup_logging(log_file: str) -> logging.Logger:
    """
    Configure structured logging for the pipeline execution.
    
    Sets up logging to write to both file and console with consistent
    formatting that includes timestamps, log levels, and messages.
    
    Args:
        log_file: Path to the log file for persistent logging
        
    Returns:
        Configured Logger instance ready for pipeline execution
    """
    # Ensure log directory exists
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure logging format with timestamp
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    
    # Create formatter
    formatter = logging.Formatter(fmt=log_format, datefmt=date_format)
    
    # Set up file handler
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    
    # Set up console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    
    # Configure root logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def load_config(config_path: str, logger: logging.Logger) -> Dict[str, Any]:
    """
    Load and validate configuration from YAML file.
    
    Reads the configuration file, extracts required parameters (seed, window, version),
    and validates their presence and types. Sets the random seed for reproducibility.
    
    Args:
        config_path: Path to the YAML configuration file
        logger: Logger instance for recording execution details
        
    Returns:
        Dictionary containing validated configuration parameters
        
    Raises:
        FileNotFoundError: If configuration file does not exist
        KeyError: If required configuration fields are missing
        ValueError: If configuration values are invalid
    """
    logger.info(f"Loading configuration from: {config_path}")
    
    # Check file existence
    if not os.path.exists(config_path):
        error_msg = f"Configuration file not found: {config_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    # Load YAML configuration
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        error_msg = f"Invalid YAML format in configuration file: {e}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Validate required fields
    required_fields = ['seed', 'window', 'version']
    missing_fields = [field for field in required_fields if field not in config]
    if missing_fields:
        error_msg = f"Missing required configuration fields: {missing_fields}"
        logger.error(error_msg)
        raise KeyError(error_msg)
    
    # Validate field types
    if not isinstance(config['seed'], int):
        error_msg = f"Configuration 'seed' must be an integer, got {type(config['seed'])}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    if not isinstance(config['window'], int) or config['window'] <= 0:
        error_msg = f"Configuration 'window' must be a positive integer, got {config['window']}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    if not isinstance(config['version'], str):
        error_msg = f"Configuration 'version' must be a string, got {type(config['version'])}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Set random seed for reproducibility
    np.random.seed(config['seed'])
    
    logger.info(f"Config loaded: seed={config['seed']}, window={config['window']}, version={config['version']}")
    
    return config


def load_data(input_path: str, logger: logging.Logger) -> pd.DataFrame:
    """
    Load and validate input CSV data.
    
    Reads the cryptocurrency OHLCV data from CSV, validates file existence,
    checks for required columns, and ensures data integrity.
    
    Args:
        input_path: Path to the input CSV file
        logger: Logger instance for recording execution details
        
    Returns:
        Pandas DataFrame containing the loaded data
        
    Raises:
        FileNotFoundError: If input file does not exist
        pd.errors.EmptyDataError: If the file is empty
        ValueError: If required columns are missing
    """
    logger.info(f"Loading data from: {input_path}")
    
    # Check file existence
    if not os.path.exists(input_path):
        error_msg = f"Input file not found: {input_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    # Load CSV data
    try:
        df = pd.read_csv(input_path)
    except pd.errors.EmptyDataError:
        error_msg = "Input file is empty or contains no parseable data"
        logger.error(error_msg)
        raise
    except pd.errors.ParserError as e:
        error_msg = f"Invalid CSV format: {e}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Validate required columns
    required_columns = ['close']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        error_msg = f"Missing required columns in dataset: {missing_columns}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Log data summary
    rows_count = len(df)
    logger.info(f"Data loaded: {rows_count} rows")
    
    # Log column information
    columns_info = ", ".join(df.columns.tolist())
    logger.debug(f"Available columns: {columns_info}")
    
    return df


def generate_signals(df: pd.DataFrame, window: int, logger: logging.Logger) -> pd.DataFrame:
    """
    Generate trading signals based on rolling mean analysis.
    
    Calculates a rolling mean on the close price column using the specified
    window size, then generates binary signals by comparing each close price
    to its corresponding rolling mean.
    
    Args:
        df: DataFrame containing at least a 'close' column
        window: Rolling window size for mean calculation
        logger: Logger instance for recording execution details
        
    Returns:
        DataFrame with additional 'rolling_mean' and 'signal' columns
        
    Raises:
        ValueError: If window parameter is invalid
    """
    logger.info(f"Calculating rolling mean with window={window}")
    
    # Validate window parameter
    if window <= 0:
        error_msg = f"Window size must be positive, got {window}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Calculate rolling mean
    # min_periods=1 ensures we get values even for early rows
    df['rolling_mean'] = df['close'].rolling(window=window, min_periods=1).mean()
    
    logger.info("Rolling mean calculated")
    
    # Generate signals: 1 if close > rolling_mean, else 0
    logger.info("Generating signals")
    df['signal'] = (df['close'] > df['rolling_mean']).astype(int)
    
    # Log signal distribution
    signal_counts = df['signal'].value_counts().to_dict()
    logger.debug(f"Signal distribution: {signal_counts}")
    
    return df


def calculate_metrics(df: pd.DataFrame, config: Dict[str, Any], 
                      start_time: float, logger: logging.Logger) -> Dict[str, Any]:
    """
    Compute and compile pipeline execution metrics.
    
    Calculates the signal rate (proportion of 1s in signals), counts processed
    rows, measures execution latency, and compiles all metrics into a structured
    dictionary suitable for JSON serialization.
    
    Args:
        df: DataFrame containing generated signals
        config: Configuration dictionary with version and seed
        start_time: Timestamp when pipeline execution started
        logger: Logger instance for recording execution details
        
    Returns:
        Dictionary containing all computed metrics
    """
    logger.info("Calculating metrics")
    
    # Calculate signal rate
    signal_rate = df['signal'].mean()
    
    # Count processed rows
    rows_processed = len(df)
    
    # Calculate execution latency in milliseconds
    latency_ms = (time.time() - start_time) * 1000
    
    # Compile metrics dictionary
    metrics = {
        "version": config['version'],
        "rows_processed": rows_processed,
        "metric": "signal_rate",
        "value": round(signal_rate, 4),
        "latency_ms": round(latency_ms, 0),
        "seed": config['seed'],
        "status": "success"
    }
    
    logger.info(f"Metrics: signal_rate={metrics['value']:.4f}, rows_processed={rows_processed}")
    
    return metrics


def write_metrics(metrics: Dict[str, Any], output_path: str, 
                  logger: logging.Logger) -> None:
    """
    Args:
        metrics: Dictionary containing computed metrics
        output_path: Path where JSON output should be written
        logger: Logger instance for recording execution details
    """
    logger.info(f"Writing metrics to: {output_path}")
    
    # Ensure output directory exists
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    # Write JSON with indentation for readability
    try:
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info("Metrics written successfully")
    except IOError as e:
        error_msg = f"Failed to write metrics file: {e}"
        logger.error(error_msg)
        raise IOError(error_msg)


def write_error_metrics(config: Dict[str, Any], error_message: str, 
                        output_path: str, logger: logging.Logger) -> None:
    """
    Args:
        config: Configuration dictionary with version
        error_message: Description of the error that occurred
        output_path: Path where error JSON should be written
        logger: Logger instance for recording execution details
    """
    logger.error(f"Writing error metrics: {error_message}")
    
    # Ensure output directory exists
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    # Create error metrics dictionary
    error_metrics = {
        "version": config.get('version', 'unknown'),
        "status": "error",
        "error_message": error_message
    }
    
    # Write error metrics
    try:
        with open(output_path, 'w') as f:
            json.dump(error_metrics, f, indent=2)
        logger.info("Error metrics written successfully")
    except IOError as e:
        logger.error(f"Failed to write error metrics file: {e}")


def main() -> int:
    """
    Main entry point for the MLOps pipeline.
    
    Orchestrates the complete pipeline execution including configuration loading,
    data ingestion, signal generation, metrics calculation, and output writing.
    Implements comprehensive error handling with graceful degradation.
    
    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    # Record pipeline start time for latency calculation
    start_time = time.time()
    
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='MLOps Pipeline: Rolling Mean Signal Generator',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input CSV file containing OHLCV data'
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to YAML configuration file'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to output JSON metrics file'
    )
    parser.add_argument(
        '--log-file',
        type=str,
        required=True,
        help='Path to log output file'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Initialize logger
    logger = setup_logging(args.log_file)
    
    logger.info("Job started")
    
    try:
        # Load configuration
        config = load_config(args.config, logger)
        
        # Load data
        df = load_data(args.input, logger)
        
        # Generate signals using rolling mean
        df = generate_signals(df, config['window'], logger)
        
        # Calculate metrics
        metrics = calculate_metrics(df, config, start_time, logger)
        
        # Write metrics to JSON
        write_metrics(metrics, args.output, logger)
        
        # Log successful completion
        latency_ms = metrics['latency_ms']
        logger.info(f"Job completed successfully in {latency_ms:.0f}ms")
        
        # Print metrics to stdout as required
        print(json.dumps(metrics, indent=2))
        
        return 0
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        write_error_metrics(
            {'version': 'unknown'},
            str(e),
            args.output,
            logger
        )
        return 1
        
    except KeyError as e:
        logger.error(f"Missing required configuration: {e}")
        write_error_metrics(
            {'version': 'unknown'},
            f"Missing required configuration: {e}",
            args.output,
            logger
        )
        return 1
        
    except ValueError as e:
        logger.error(f"Invalid configuration or data: {e}")
        write_error_metrics(
            {'version': 'unknown'},
            f"Invalid configuration or data: {e}",
            args.output,
            logger
        )
        return 1
        
    except Exception as e:
        logger.error(f"Unexpected error during pipeline execution: {e}")
        write_error_metrics(
            {'version': 'unknown'},
            f"Unexpected error: {e}",
            args.output,
            logger
        )
        return 1


if __name__ == '__main__':
    sys.exit(main())
