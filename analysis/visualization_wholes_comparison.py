# Model Comparison Visualization
# This script compares the performance of unimodal and multimodal deep learning models
# across different datasets, input lengths, and output lengths.

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.data import experiment

# Set plot style
# plt.style.use('seaborn-whitegrid')
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.2)
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi'] = 100


def load_data():
    """Load the data from CSV files"""
    # Load the unimodal results
    unimodal_df = pd.read_csv('../benchmarks/logs/unimodal_result.csv')

    # Load the multimodal results
    multimodal_df = pd.read_csv('../logs/prior_adding/tff.csv')

    # Load the tsflib results
    tsflib_df = pd.read_csv('../benchmarks/tsflib/w_i_prior0.5.csv')

    print("Unimodal Results:")
    print(unimodal_df.head())

    print("\nMultimodal Results:")
    print(multimodal_df.head())

    print("\nTSFlib Results:")
    print(tsflib_df.head())

    return unimodal_df, multimodal_df, tsflib_df


def explore_data(unimodal_df, multimodal_df, tsflib_df):
    """Explore the structure of the datasets"""
    print("Unimodal Dataset Info:")
    print(f"Shape: {unimodal_df.shape}")
    print(f"Datasets: {unimodal_df['dataset'].unique()}")
    print(f"Models: {unimodal_df['model'].unique()}")
    print(f"Input lengths: {sorted(unimodal_df['input_len'].unique())}")
    print(f"Output lengths: {sorted(unimodal_df['output_len'].unique())}")

    print("\nMultimodal Dataset Info:")
    print(f"Shape: {multimodal_df.shape}")
    print(f"Datasets: {multimodal_df['dataset'].unique()}")
    print(f"Input lengths: {sorted(multimodal_df['input_length'].unique())}")
    print(f"Output lengths: {sorted(multimodal_df['prediction_length'].unique())}")

    print("\nTSFlib Dataset Info:")
    print(f"Shape: {tsflib_df.shape}")
    print(f"Datasets: {tsflib_df['dataset'].unique()}")
    print(f"Models: {tsflib_df['model'].unique()}")
    print(f"LLM Models: {tsflib_df['llm_model'].unique()}")
    print(f"Input lengths: {sorted(tsflib_df['input_len'].unique())}")
    print(f"Output lengths: {sorted(tsflib_df['output_len'].unique())}")


def prepare_data(unimodal_df, multimodal_df, tsflib_df):
    """Prepare the data for comparison"""
    # Rename columns in multimodal_df to match unimodal_df
    multimodal_df = multimodal_df.rename(columns={
        'input_length': 'input_len',
        'prediction_length': 'output_len'
    })

    # Add a model_type column to distinguish between unimodal, multimodal, and tsflib
    unimodal_df['model_type'] = 'Unimodal'
    multimodal_df['model_type'] = 'Multimodal'
    tsflib_df['model_type'] = 'TSFlib'

    # For multimodal_df, set a consistent model name
    multimodal_df['model'] = 'iTransformer_GPT2'

    # For tsflib_df, create a combined model name using model and llm_model
    tsflib_df['model'] = tsflib_df['model'] + '_' + tsflib_df['llm_model']

    # Select common columns for comparison
    common_columns = ['dataset', 'input_len', 'output_len', 'model', 'model_type', 'mse', 'mae', 'rmse', 'mape', 'mspe']

    # Create a combined dataframe for easier comparison
    combined_df = pd.concat([
        unimodal_df[common_columns], 
        multimodal_df[common_columns], 
        tsflib_df[common_columns]
    ], ignore_index=True)

    print("Combined DataFrame:")
    print(combined_df.head())

    return combined_df


def plot_metric_by_dataset(df, metric, title=None, save_path=None):
    """Create comparison plots for a specific metric by dataset"""
    plt.figure(figsize=(14, 10))

    # Group by dataset and model_type, and calculate the mean of the metric
    grouped = df.groupby(['dataset', 'model_type'])[metric].mean().reset_index()

    # Create a bar plot
    ax = sns.barplot(x='dataset', y=metric, hue='model_type', data=grouped)

    # Add labels and title
    plt.xlabel('Dataset')
    plt.ylabel(metric.upper())
    plt.title(title or f'Average {metric.upper()} by Dataset')
    plt.xticks(rotation=45)

    # Add value labels on top of bars
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}',
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom',
                    xytext=(0, 5), textcoords='offset points')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def plot_metric_by_input_len(df, metric, title=None, save_path=None):
    """Create comparison plots for a specific metric by input length"""
    plt.figure(figsize=(14, 8))

    # Group by input_len and model_type, and calculate the mean of the metric
    grouped = df.groupby(['input_len', 'model_type'])[metric].mean().reset_index()

    # Create a bar plot
    ax = sns.barplot(x='input_len', y=metric, hue='model_type', data=grouped)

    # Add labels and title
    plt.xlabel('Input Length')
    plt.ylabel(metric.upper())
    plt.title(title or f'Average {metric.upper()} by Input Length')

    # Add value labels on top of bars
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}',
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom',
                    xytext=(0, 5), textcoords='offset points')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def plot_metric_by_output_len(df, metric, title=None, save_path=None):
    """Create comparison plots for a specific metric by output length"""
    plt.figure(figsize=(14, 8))

    # Group by output_len and model_type, and calculate the mean of the metric
    grouped = df.groupby(['output_len', 'model_type'])[metric].mean().reset_index()

    # Create a bar plot
    ax = sns.barplot(x='output_len', y=metric, hue='model_type', data=grouped)

    # Add labels and title
    plt.xlabel('Output Length')
    plt.ylabel(metric.upper())
    plt.title(title or f'Average {metric.upper()} by Output Length')

    # Add value labels on top of bars
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}',
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom',
                    xytext=(0, 5), textcoords='offset points')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def plot_heatmaps_by_dataset(df, save_dir=None):
    """Create heatmaps showing performance by input and output length for each dataset"""
    # Get unique datasets
    datasets = df['dataset'].unique()

    # For each dataset, create a heatmap showing performance by input and output length
    for dataset in datasets:
        # Filter data for the current dataset
        dataset_df = df[df['dataset'] == dataset]

        # Create a figure with subplots for unimodal, multimodal, and tsflib
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))
        fig.suptitle(f'RMSE Comparison for {dataset} Dataset', fontsize=16)

        # Unimodal plot
        unimodal_data = dataset_df[dataset_df['model_type'] == 'Unimodal']
        if not unimodal_data.empty:
            # Get the best model for each input/output length combination
            best_unimodal = unimodal_data.groupby(['input_len', 'output_len'])['rmse'].min().reset_index()
            # Create a pivot table for the heatmap
            pivot_uni = best_unimodal.pivot(index='input_len', columns='output_len', values='rmse')
            # Plot the heatmap
            sns.heatmap(pivot_uni, annot=True, fmt='.2f', cmap='YlGnBu', ax=axes[0])
            axes[0].set_title('Best Unimodal Model RMSE')
            axes[0].set_xlabel('Output Length')
            axes[0].set_ylabel('Input Length')
        else:
            axes[0].text(0.5, 0.5, 'No unimodal data available', ha='center', va='center')
            axes[0].set_title('Unimodal Model (No Data)')

        # Multimodal plot
        multimodal_data = dataset_df[dataset_df['model_type'] == 'Multimodal']
        if not multimodal_data.empty:
            # Create a pivot table for the heatmap
            pivot_multi = multimodal_data.pivot(index='input_len', columns='output_len', values='rmse')
            # Plot the heatmap
            sns.heatmap(pivot_multi, annot=True, fmt='.2f', cmap='YlGnBu', ax=axes[1])
            axes[1].set_title('Multimodal Model RMSE')
            axes[1].set_xlabel('Output Length')
            axes[1].set_ylabel('Input Length')
        else:
            axes[1].text(0.5, 0.5, 'No multimodal data available', ha='center', va='center')
            axes[1].set_title('Multimodal Model (No Data)')

        # TSFlib plot
        tsflib_data = dataset_df[dataset_df['model_type'] == 'TSFlib']
        if not tsflib_data.empty:
            # Create a pivot table for the heatmap
            pivot_tsf = tsflib_data.pivot(index='input_len', columns='output_len', values='rmse')
            # Plot the heatmap
            sns.heatmap(pivot_tsf, annot=True, fmt='.2f', cmap='YlGnBu', ax=axes[2])
            axes[2].set_title('TSFlib Model RMSE')
            axes[2].set_xlabel('Output Length')
            axes[2].set_ylabel('Input Length')
        else:
            axes[2].text(0.5, 0.5, 'No TSFlib data available', ha='center', va='center')
            axes[2].set_title('TSFlib Model (No Data)')

        plt.tight_layout(rect=[0, 0, 1, 0.95])

        if save_dir:
            plt.savefig(f"{save_dir}/heatmap_{dataset}.png")
        else:
            plt.show()


def compare_best_models(df, save_dir=None):
    """Compare the best unimodal, multimodal, and tsflib models for each dataset"""

    # Function to find the best model for each dataset based on RMSE
    def get_best_models(df):
        # For unimodal models, find the best model for each dataset
        best_unimodal = df[df['model_type'] == 'Unimodal'].groupby('dataset')['rmse'].min().reset_index()
        best_unimodal = best_unimodal.rename(columns={'rmse': 'best_unimodal_rmse'})

        # For multimodal models, find the best model for each dataset
        best_multimodal = df[df['model_type'] == 'Multimodal'].groupby('dataset')['rmse'].min().reset_index()
        best_multimodal = best_multimodal.rename(columns={'rmse': 'best_multimodal_rmse'})

        # For tsflib models, find the best model for each dataset
        best_tsflib = df[df['model_type'] == 'TSFlib'].groupby('dataset')['rmse'].min().reset_index()
        best_tsflib = best_tsflib.rename(columns={'rmse': 'best_tsflib_rmse'})

        # Merge the results
        best_models = pd.merge(best_unimodal, best_multimodal, on='dataset', how='outer')
        best_models = pd.merge(best_models, best_tsflib, on='dataset', how='outer')

        # Calculate the improvement percentage of multimodal over unimodal
        best_models['multimodal_improvement'] = ((best_models['best_unimodal_rmse'] - best_models['best_multimodal_rmse']) /
                                      best_models['best_unimodal_rmse'] * 100)

        # Calculate the improvement percentage of tsflib over unimodal
        best_models['tsflib_improvement'] = ((best_models['best_unimodal_rmse'] - best_models['best_tsflib_rmse']) /
                                      best_models['best_unimodal_rmse'] * 100)

        return best_models

    # Get the best models
    best_models = get_best_models(df)

    # Display the results
    print("Best Models Comparison:")
    print(best_models)

    # Plot the comparison of best models
    plt.figure(figsize=(14, 8))

    # Create a bar plot
    x = np.arange(len(best_models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 8))
    rects1 = ax.bar(x - width, best_models['best_unimodal_rmse'], width, label='Best Unimodal')
    rects2 = ax.bar(x, best_models['best_multimodal_rmse'], width, label='Multimodal')
    rects3 = ax.bar(x + width, best_models['best_tsflib_rmse'], width, label='TSFlib')

    # Add labels and title
    ax.set_xlabel('Dataset')
    ax.set_ylabel('RMSE (lower is better)')
    ax.set_title('Comparison of Best Models by Dataset')
    ax.set_xticks(x)
    ax.set_xticklabels(best_models['dataset'], rotation=45)
    ax.legend()

    # Add value labels on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    plt.tight_layout()

    if save_dir:
        plt.savefig(f"{save_dir}/best_models_comparison.png")
    else:
        plt.show()

    # Plot the improvement percentage for multimodal
    plt.figure(figsize=(14, 8))
    bars = plt.bar(best_models['dataset'], best_models['multimodal_improvement'])

    # Color the bars based on whether the improvement is positive or negative
    for i, bar in enumerate(bars):
        if best_models['multimodal_improvement'].iloc[i] > 0:
            bar.set_color('green')
        else:
            bar.set_color('red')

    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.xlabel('Dataset')
    plt.ylabel('Improvement Percentage (%)')
    plt.title('Multimodal Model Improvement Over Best Unimodal Model')
    plt.xticks(rotation=45)

    # Add value labels on top of bars
    for i, v in enumerate(best_models['multimodal_improvement']):
        plt.text(i, v + (5 if v > 0 else -5), f'{v:.1f}%', ha='center', va='bottom' if v > 0 else 'top')

    plt.tight_layout()

    if save_dir:
        plt.savefig(f"{save_dir}/multimodal_improvement_percentage.png")
    else:
        plt.show()

    # Plot the improvement percentage for tsflib
    plt.figure(figsize=(14, 8))
    bars = plt.bar(best_models['dataset'], best_models['tsflib_improvement'])

    # Color the bars based on whether the improvement is positive or negative
    for i, bar in enumerate(bars):
        if best_models['tsflib_improvement'].iloc[i] > 0:
            bar.set_color('green')
        else:
            bar.set_color('red')

    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.xlabel('Dataset')
    plt.ylabel('Improvement Percentage (%)')
    plt.title('TSFlib Model Improvement Over Best Unimodal Model')
    plt.xticks(rotation=45)

    # Add value labels on top of bars
    for i, v in enumerate(best_models['tsflib_improvement']):
        plt.text(i, v + (5 if v > 0 else -5), f'{v:.1f}%', ha='center', va='bottom' if v > 0 else 'top')

    plt.tight_layout()

    if save_dir:
        plt.savefig(f"{save_dir}/tsflib_improvement_percentage.png")
    else:
        plt.show()


def compare_models_for_dataset(df, save_dir=None):
    # Function to compare models for a specific datasets combination
    def compare_models_for_datasets(df, datasets, metric='mse'):
        # Filter data for the specified datasets
        filtered_df = df[df['dataset'].isin(datasets)]

        input_len = filtered_df['input_len'].unique()
        output_len = filtered_df['output_len'].unique()

        if filtered_df.empty:
            print(f"No data available for datasets={datasets}")
            return

        # For each input/output length combination, plot the performance of each model
        for dataset in datasets:
            dataset_df = filtered_df[filtered_df['dataset'] == dataset]
            if dataset_df.empty:
                print(f"No data available for dataset={dataset}")
                continue

            for (input_len, output_len) in dataset_df[['input_len', 'output_len']].drop_duplicates().values:
                plt.figure(figsize=(14, 8))
                config_df = dataset_df[
                    (dataset_df['input_len'] == input_len) & (dataset_df['output_len'] == output_len)]

                # Get unimodal and tsflib data
                unimodal_data = config_df[config_df['model_type'] == 'Unimodal'].sort_values(metric)
                tsflib_data = config_df[config_df['model_type'] == 'TSFlib'].sort_values(metric)

                # Set up bar width and positions
                bar_width = 0.35
                unimodal_positions = []
                tsflib_positions = []
                all_models = []

                # Calculate positions for bars
                if not unimodal_data.empty and not tsflib_data.empty:
                    # Both unimodal and tsflib data exist
                    total_models = len(unimodal_data) + len(tsflib_data)
                    x_positions = np.arange(total_models)
                    unimodal_positions = x_positions[:len(unimodal_data)]
                    tsflib_positions = x_positions[len(unimodal_data):]
                    all_models = list(unimodal_data['model']) + list(tsflib_data['model'])
                elif not unimodal_data.empty:
                    # Only unimodal data exists
                    unimodal_positions = np.arange(len(unimodal_data))
                    all_models = list(unimodal_data['model'])
                elif not tsflib_data.empty:
                    # Only tsflib data exists
                    tsflib_positions = np.arange(len(tsflib_data))
                    all_models = list(tsflib_data['model'])

                # Create a bar plot for unimodal models
                if not unimodal_data.empty:
                    bars1 = plt.bar(unimodal_positions, unimodal_data[metric], width=bar_width,
                                    label=f'Unimodal Models (IL={input_len}, OL={output_len})')

                    # Add value labels on top of bars
                    for bar in bars1:
                        height = bar.get_height()
                        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                                 f'{height:.2f}', ha='center', va='bottom')

                # Create a bar plot for tsflib models
                if not tsflib_data.empty:
                    bars2 = plt.bar(tsflib_positions, tsflib_data[metric], width=bar_width,
                                    label=f'TSFlib Models (IL={input_len}, OL={output_len})')

                    # Add value labels on top of bars
                    for bar in bars2:
                        height = bar.get_height()
                        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                                 f'{height:.2f}', ha='center', va='bottom')

                # Add x-tick labels for all models
                plt.xticks(np.arange(len(all_models)), all_models, rotation=45, ha='right')

                # Add a horizontal line for the multimodal model
                multimodal_data = config_df[config_df['model_type'] == 'Multimodal']
                if not multimodal_data.empty:
                    multimodal_value = multimodal_data[metric].values[0]
                    plt.axhline(y=multimodal_value, linestyle='--',
                                label=f'Multimodal (IL={input_len}, OL={output_len}): {multimodal_value:.2f}')

                plt.xlabel('Model')
                plt.ylabel(metric.upper())
                plt.title(
                    f'{metric.upper()} Comparison for {dataset} (Input Length: {input_len}, Output Length: {output_len})')
                plt.legend()
                plt.tight_layout()
                if save_dir:
                    plt.savefig(f"{save_dir}/comparison_{dataset}_il{input_len}_ol{output_len}.png")
                else:
                    plt.show()

    # Compare models for specific datasets
    datasets = df['dataset'].unique()
    compare_models_for_datasets(df, datasets=datasets, metric='mse')

    # # Compare models for some common configurations
    # common_configs = [
    #     (8, 6),
    #     (8, 12),
    #     (24, 6),
    #     (24, 12)
    # ]
    #
    # for input_len, output_len in common_configs:
    #     compare_models_for_config(df, input_len, output_len)


def main():
    """Main function to run all visualizations"""
    # Load data
    unimodal_df, multimodal_df, tsflib_df = load_data()

    # Explore data
    explore_data(unimodal_df, multimodal_df, tsflib_df)

    # Prepare data for comparison
    combined_df = prepare_data(unimodal_df, multimodal_df, tsflib_df)
    # remove temporary dataset Security
    combined_df = combined_df[combined_df['dataset'] != 'Agriculture']
    # remove temporary dataset SocialGood
    # combined_df = combined_df[combined_df['dataset'] != 'SocialGood']

    # Create visualizations
    # 1. Performance comparison by dataset
    for metric in ['mse', 'mae']: #, 'rmse', 'mape', 'mspe']:
        plot_metric_by_dataset(combined_df, metric)

    # 2. Performance comparison by input length
    for metric in ['mse', 'mae', 'rmse']:
        plot_metric_by_input_len(combined_df, metric)

    # 3. Performance comparison by output length
    for metric in ['mse', 'mae', 'rmse']:
        plot_metric_by_output_len(combined_df, metric)

    # 4. Detailed comparison by dataset and input/output length
    plot_heatmaps_by_dataset(combined_df)

    # 5. Performance comparison of best models
    compare_best_models(combined_df)

    # 6. Detailed model comparison for specific input/output length combinations
    compare_models_for_dataset(combined_df)

    print("Visualization complete!")


if __name__ == "__main__":
    main()
