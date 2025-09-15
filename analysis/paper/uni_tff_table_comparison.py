# Table Comparison for Academic Paper
# This script generates comparison tables between TFF model and univariable models
# across different datasets, suitable for inclusion in academic papers.

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

exp_settings_filter = {
    'Agriculture': {
        'input_length': 24,
        'prediction_length': [6, 8, 10, 12]
    },
    'Climate': {
        'input_length': 8,
        'prediction_length': [6, 8, 10, 12]
    },
    'Economy': {
        'input_length': 24,
        'prediction_length': [6, 8, 10, 12]
    },
    'Energy': {
        'input_length': 24,
        'prediction_length': [12, 24, 36, 48]
    },
    'Environment': {
        'input_length': 84,
        'prediction_length': [24, 96, 192, 336]
    },
    'Health': {
        'input_length': 36,
        'prediction_length': [12, 24, 36, 48]
    },
    'Security': {
        'input_length': 8,
        'prediction_length': [6, 8, 10, 12]
    },
    'SocialGood': {
        'input_length': 24,
        'prediction_length': [6, 8, 10, 12]
    },
    'Traffic': {
        'input_length': 8,
        'prediction_length': [6, 8, 10, 12]
    }
}

metrics = ['mse', 'mae']
unimodel_names = ['iTransformer', 'PatchTST', 'Crossformer', 'Autoformer', 'Informer']


def load_data():
    """Load the data from CSV files"""
    # Get the project root directory
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    # Load the unimodal results
    unimodal_path = project_root / 'benchmarks' / 'logs' / 'unimodal_result.csv'
    unimodal_df = pd.read_csv(unimodal_path)

    # Load the TFF results
    tff_path = project_root / 'logs' / 'prior_added_at_fusion' / 'tff.csv'
    tff_df = pd.read_csv(tff_path)

    print("Unimodal Results:")
    print(unimodal_df.head())

    print("\nTFF Results:")
    print(tff_df.head())

    return unimodal_df, tff_df


def prepare_data(unimodal_df, tff_df):
    """Prepare the data for comparison"""
    # Add a model_type column to distinguish between unimodal and TFF
    unimodal_df['model_type'] = 'Unimodal'
    tff_df['model_type'] = 'TFF'

    # For tff_df, set a consistent model name
    tff_df['model'] = 'TFF'

    # Select common columns for comparison
    common_columns = ['dataset', 'input_length', 'prediction_length', 'model', 'model_type', 'mse', 'mae', 'rmse',
                      'mape', 'mspe']

    # Create a combined dataframe for easier comparison
    combined_df = pd.concat([
        unimodal_df[common_columns],
        tff_df[common_columns]
    ], ignore_index=True)

    # skip dataset name Security, Agriculture, Energy
    # combined_df = combined_df[combined_df['dataset'] != 'Security']
    # combined_df = combined_df[combined_df['dataset'] != 'Agriculture']
    # combined_df = combined_df[combined_df['dataset'] != 'Energy']

    filtered_df = combined_df[combined_df.apply(
        lambda row: (
                row['dataset'] in exp_settings_filter and
                row['input_length'] == exp_settings_filter[row['dataset']]['input_length'] and
                row['prediction_length'] in exp_settings_filter[row['dataset']]['prediction_length']
        ), axis=1
    )]

    print("Combined DataFrame:")
    print(filtered_df.head())

    return filtered_df


def generate_detailed_comparison_tables(df, save_dir=None):
    """Generate detailed comparison tables for each dataset and input/output length combination"""
    # Get unique datasets
    datasets = df['dataset'].unique()

    # Dictionary to store all tables
    all_tables = {}

    columns = ['Experiments']  # this is the template columns for each table
    for model_name in ['TTF'] + unimodel_names:
        for metric in metrics:
            columns.append(f"{model_name}_{metric}")

    # For each dataset, create a comparison table
    for dataset in datasets:
        # Filter data for the current dataset
        dataset_df = df[df['dataset'] == dataset]

        if dataset_df.empty:
            print(f"No data available for dataset={dataset}")
            continue

        table_dataset = pd.DataFrame(columns=columns)

        # Get unique input/output length combinations for this dataset
        configs = dataset_df[['input_length', 'prediction_length']].drop_duplicates().values

        for (input_len, output_len) in configs:
            # Filter data for the current input/output length combination
            config_df = dataset_df[
                (dataset_df['input_length'] == input_len) &
                (dataset_df['prediction_length'] == output_len)
                ]

            data_row = [input_len]

            # Get TFF data
            tff_data = config_df[config_df['model_type'] == 'TFF']

            for metric in metrics:
                if not tff_data.empty:
                    data_row.append(round(tff_data.iloc[0][metric], 3))
                else:
                    data_row.append(np.nan)

            # Get univariable data
            # univariable_data = config_df[config_df['model_type'] == 'Unimodal'].sort_values('rmse')
            univariable_data = None
            for model_name in unimodel_names:
                univariable_data = config_df[
                    (config_df['model_type'] == 'Unimodal') & (config_df['model'] == model_name)]
                for metric in metrics:
                    if not univariable_data.empty:
                        data_row.append(round(univariable_data.iloc[0][metric], 4))
                    else:
                        data_row.append(np.nan)

            # Adding new row to the table
            table_dataset.loc[len(table_dataset)] = data_row

        # Store the table
        table_key = f"{dataset}"
        all_tables[table_key] = table_dataset

        # Save the table
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

            # Save as CSV
            csv_path = f"{save_dir}/{table_key}_comparison.csv"
            table_dataset.to_csv(csv_path, index=False)
    return all_tables


def generate_aggregate_comparison_table(df, save_dir=None):
    """Generate an aggregate comparison table across all datasets"""
    # Group by dataset and model_type, and calculate the mean of each metric
    grouped_df = df.groupby(['dataset', 'model_type', 'model'])[metrics].mean().reset_index()

    # Create a pivot table with datasets as rows and model_types as columns
    ranking_df = pd.DataFrame(columns=['Domain', 'TFF_mse', 'TFF_mae', 'Best_baseline_mse', 'Best_baseline_mae',
                                             'Improvement_mse (%)', 'Improvement_mae (%)',
                                             'Rank_mse', 'Rank_mae'])

    for dataset in grouped_df['dataset'].unique():
        unimodal_data = grouped_df[(grouped_df['dataset'] == dataset) & (grouped_df['model_type'] == 'Unimodal')]
        tff_data = grouped_df[(grouped_df['dataset'] == dataset) & (grouped_df['model_type'] == 'TFF')].iloc[0]
        # sort unimodal data by mse
        unimodal_data = unimodal_data.sort_values(by='mse')
        if unimodal_data.empty or tff_data.empty:
            continue

        best_baseline = unimodal_data.iloc[0]
        improvement_mse = ((best_baseline['mse'] - tff_data['mse']) / best_baseline['mse'] * 100).round(2)
        improvement_mae = ((best_baseline['mae'] - tff_data['mae']) / best_baseline['mae'] * 100).round(2)
        rank_mse = (unimodal_data['mse'] < tff_data['mse']).sum() + 1
        rank_mae = (unimodal_data['mae'] < tff_data['mae']).sum() + 1
        ranking_df.loc[len(ranking_df)] = [
            dataset,
            round(tff_data['mse'], 3),
            round(tff_data['mae'], 3),
            round(best_baseline['mse'], 3),
            round(best_baseline['mae'], 3),
            improvement_mse,
            improvement_mae,
            rank_mse,
            rank_mae
        ]



    # Save the table
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

        # Save as CSV
        csv_path = f"{save_dir}/summary_performance_ranking.csv"
        ranking_df.to_csv(csv_path, index=False)
        print(f"Aggregate comparison table saved to: {csv_path}")

    return ranking_df


def main():
    """Main function to generate comparison tables"""
    # Get the project root directory
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    # Load data
    unimodal_df, tff_df = load_data()

    # Prepare data for comparison
    combined_df = prepare_data(unimodal_df, tff_df)

    # Create output directory
    output_dir = project_root / "results" / "table_comparison"
    os.makedirs(output_dir, exist_ok=True)

    # Generate detailed comparison tables
    detailed_tables = generate_detailed_comparison_tables(combined_df, save_dir=output_dir)

    # Generate aggregate comparison tables
    aggregate_tables = generate_aggregate_comparison_table(combined_df, save_dir=output_dir)

    print(f"Table generation complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
