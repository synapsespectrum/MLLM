"""
Fusion Cross Attention Results Gatherer

This script automatically gathers results from multiple text files in the logs\fusion_cross_attention directory
and compiles them into a single CSV file. Each text file contains experiment results with metrics like mse, mae, rmse, etc.
If a file has multiple results, the script selects the best one based on MSE (lower is better).

The script extracts:
- Dataset name (from directory name)
- Input length (from filename)
- Prediction length (from filename)
- Model name (from result line)
- Metrics (mse, mae, rmse, mape, mspe)

Usage:
    python gather_results.py

Output:
    fusion_cross_attention_results.csv
"""

import os
import re
import csv
import glob
import argparse



def parse_metrics(metrics_line):
    """
    Parse metrics from a line of text.

    Args:
        metrics_line (str): Line containing metrics in format "key1:value1, key2:value2, ..."

    Returns:
        dict: Dictionary of metrics with keys as metric names and values as float values
    """
    metrics = {}
    for metric in metrics_line.split(', '):
        key, value = metric.split(':')
        metrics[key] = float(value)
    return metrics


def extract_il_pl(filename):
    """
    Extract input length and prediction length from filename.

    Args:
        filename (str): Filename in format "results_il{input_length}_pl{prediction_length}.txt"

    Returns:
        tuple: (input_length, prediction_length) as integers, or (None, None) if not found
    """
    match = re.search(r'results_il(\d+)_pl(\d+)\.txt', filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None


def extract_model_name(line):
    """
    Extract model name from result line.

    Args:
        line (str): Line containing model name in format "[timestamp] model_name_il..."

    Returns:
        str: Model name, or "Unknown" if not found
    """
    # Extract the text between timestamp and _il
    match = re.search(r'\] (.*?)_il', line)
    if match:
        return match.group(1)
    return "Unknown"


def process_result_file(file_path):
    """
    Process a single result file and return the best result based on MSE.

    Args:
        file_path (str): Path to the result file

    Returns:
        tuple: (timestamp_line, metrics_line) of the best result, or None if no results found
    """
    with open(file_path, 'r') as f:
        content = f.read()

    # Split content into result blocks (timestamp line + metrics line)
    result_blocks = []
    lines = content.strip().split('\n')

    i = 0
    while i < len(lines):
        if lines[i].startswith('[') and i + 1 < len(lines) and lines[i + 1].startswith('mse:'):
            timestamp_line = lines[i]
            metrics_line = lines[i + 1]
            result_blocks.append((timestamp_line, metrics_line))
            i += 2
        else:
            i += 1

    if not result_blocks:
        return None

    # Find the result with the lowest MSE
    best_result = min(result_blocks, key=lambda x: parse_metrics(x[1])['mse'])

    return best_result

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_dir", type=str, default="fusion_cross_attention", help="Experiment directory")
    args = parser.parse_args()
    return args

def main():
    """
    Main function to gather results from all result files and save them to a CSV file.
    """
    args = parse_args()
    base_dir = args.experiment_dir
    output_file = os.path.join(base_dir, 'ttf.csv')

    # Define CSV headers
    headers = ['dataset', 'model', 'input_length', 'prediction_length', 'mse', 'mae', 'rmse', 'mape', 'mspe']

    results = []

    # Traverse through all subdirectories
    for dataset_dir in os.listdir(base_dir):
        dataset_path = os.path.join(base_dir, dataset_dir)

        if not os.path.isdir(dataset_path):
            continue

        print(f"Processing dataset: {dataset_dir}")

        # Process all result files in the dataset directory
        for result_file in glob.glob(os.path.join(dataset_path, 'results_il*_pl*.txt')):
            input_length, prediction_length = extract_il_pl(os.path.basename(result_file))

            if input_length is None or prediction_length is None:
                continue

            print(f"  Processing file: {os.path.basename(result_file)}")

            best_result = process_result_file(result_file)

            if best_result:
                timestamp_line, metrics_line = best_result
                model = extract_model_name(timestamp_line)
                metrics = parse_metrics(metrics_line)

                results.append({
                    'dataset': dataset_dir,
                    'model': model,
                    'input_length': input_length,
                    'prediction_length': prediction_length,
                    'mse': metrics.get('mse', ''),
                    'mae': metrics.get('mae', ''),
                    'rmse': metrics.get('rmse', ''),
                    'mape': metrics.get('mape', ''),
                    'mspe': metrics.get('mspe', '')
                })

    # Write results to CSV
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(results)

    print(f"Results have been saved to {output_file}")
    print(f"Total results gathered: {len(results)}")


if __name__ == "__main__":
    main()
