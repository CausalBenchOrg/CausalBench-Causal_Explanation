import atexit
from collections import defaultdict
import shutil
import os
import tempfile
from urllib.parse import urlparse
import pandas as pd
import numpy as np
import requests
import networkx as nx
from dowhy import CausalModel
from common.common_constants import RANDOM_SEED, TEMP_DIR
from common.yaml_to_csv import main as process_yaml_data, headers
from sklearn.preprocessing import LabelEncoder, StandardScaler


os.environ['MPLCONFIGDIR'] = f'{TEMP_DIR}/mplconfig'

def compute_CATE(data, treatment, outcome, graph):
    try:
        data_clean = data.copy()
        
        np.random.seed(RANDOM_SEED)
        
        if treatment in data_clean.columns:
            data_clean[treatment] = pd.to_numeric(data_clean[treatment], errors='coerce')
        if outcome in data_clean.columns:
            data_clean[outcome] = pd.to_numeric(data_clean[outcome], errors='coerce')
        
        data_clean = data_clean.dropna(subset=[treatment, outcome])
        
        data_clean = data_clean.sort_values([treatment, outcome]).reset_index(drop=True)
        
        if len(data_clean) < 3:
            # return np.nan
            return 0.0  # return 0 instead of nan
            
        if data_clean[treatment].std() == 0 or data_clean[outcome].std() == 0:
            return 0.0
        
        model = CausalModel(
            data=data_clean,
            treatment=treatment,
            outcome=outcome,
            graph=graph
        )

        identified_estimand = model.identify_effect()

        estimate = model.estimate_effect(
            identified_estimand,
            method_name='backdoor.linear_regression',
            test_significance=True
        )

        return estimate.value
    except Exception as e:
        print(f"Error in CATE computation for {treatment} -> {outcome}: {e}")
        # return np.nan
        return 0.0  # return 0 instead of nan


def compute_score(data, hyperparameters, outcome_column):
    data = data.copy()
    
    cols_to_drop = []
    for col in data.columns:
        if col not in hyperparameters and col != outcome_column:
            cols_to_drop.append(col)
    
    if cols_to_drop:
        data = data.drop(columns=cols_to_drop)

    G = nx.DiGraph()
    for hyperparameter in sorted(hyperparameters):
        G.add_edge(hyperparameter, outcome_column)

    scores = np.zeros(shape=(len(hyperparameters), 1))
    scores = pd.DataFrame(scores, index=sorted(hyperparameters), columns=[outcome_column])

    for hyperparameter in sorted(hyperparameters):
        scores.loc[hyperparameter, outcome_column] = compute_CATE(data, hyperparameter, outcome_column, G)

    return scores


def run_causal_analysis(download_dir,
                        hp_dtypes=None,
                        candidate_hyperparameters=None, 
                        outcome_column=None,
                        logger=None):
    """
    Run causal analysis on data from the provided ZIP URLs.
    
    Args:
        download_dir (string): Path to directory containing ZIP files to analyze
        candidate_hyperparameters (list): List of hyperparameter column names to analyze
        outcome_column (str): Column name for the outcome variable to analyze
        output_filename (str): Name of the output YAML file
    
    Returns:
        dict: Analysis results
    """
    group_by_metric = False
    if outcome_column is None:
        outcome_column = 'Time.Duration'
    
    if outcome_column.startswith('Metric.Score') and not group_by_metric:
        group_by_metric = True
        print(f"Auto-enabling metric grouping for outcome: {outcome_column}")
    
    outcome_column_mapping = {
        'Time.Duration': ['DS.Rows', 'DS.Cols', 'HW.', 'SW.', 'HP.', 'Model.'],
        'Metric.Score': ['HP.'] if group_by_metric else ['DS.Rows', 'DS.Cols', 'SW.', 'HP.', 'Model.'],
        'Metric.GPUMemoryIdle':['DS.Rows', 'DS.Cols', 'SW.', 'HP.', 'Model.'],
        'Metric.GPUMemoryPeak':['DS.Rows', 'DS.Cols', 'SW.', 'HP.', 'Model.'],
        'Metric.ReadBytes':['DS.Rows', 'DS.Cols', 'SW.', 'HP.', 'Model.'],
        'Metric.WriteBytes':['DS.Rows', 'DS.Cols', 'SW.', 'HP.', 'Model.'],
        'Metric.Memory':['DS.Rows', 'DS.Cols', 'SW.', 'HP.', 'Model.']
    }
    
    # if candidate_hyperparameters is None:
    #     if outcome_column in outcome_column_mapping:
    #         candidate_hyperparameters = None
    #     else:
    #         print("No candidate hyperparameters provided and no default mapping found")
    
    encode = []
    try:    
        if download_dir:
            print(f"Processing ZIP files from {download_dir}")
            raw_df = process_yaml_data(download_dir, headers)
        else:
            print(f"Invalid location: {download_dir}")

        raw_df = raw_df.sort_values(['DS.Name'] + [col for col in sorted(raw_df.columns) if col != 'DS.Name']).reset_index(drop=True)

        if outcome_column in outcome_column_mapping:
            prefixes = outcome_column_mapping[outcome_column]
            potential_cols = [col for col in raw_df.columns 
                            if any(col.startswith(prefix) for prefix in prefixes)]
            
            candidate_hyperparameters = []
            for col in potential_cols:
                hp_col = col.split(".")[1]  # Remove 'HP.' prefix
                if hp_col in hp_dtypes and hp_dtypes[hp_col] in ['integer', 'decimal']:
                    candidate_hyperparameters.append(col)
                else:
                    print(f"Skipping non-numeric column: {col}")
            
            print(f"Auto-selected candidate_hyperparameters for {outcome_column}: {len(candidate_hyperparameters)} numeric columns with prefixes {prefixes}")
        
        elif candidate_hyperparameters is None:
            print("No candidate hyperparameters provided and no default mapping found")
        
        hw_cols = [col for col in raw_df.columns if any(hw in col for hw in ['HW.', 'Model.', 'Time.', 'SW.', 'HP.', 'Metric.'])]
        print(f"Available columns: {sorted(hw_cols)}")
        print(f"Datasets: {sorted(raw_df['DS.Name'].unique())}")
        
        hyperparameters = []
        for feature in sorted(candidate_hyperparameters):
            if feature in raw_df.columns:
                unique_values = raw_df[feature].dropna().unique()
                if len(unique_values) > 1: 
                    hyperparameters.append(feature)
                    print(f"{feature}: {len(unique_values)} unique values")
                else:
                    print(f"{feature}: Only {len(unique_values)} unique value(s) - skipping")
            else:
                print(f"{feature}: Not found in data - skipping")
        
        hyperparameters = sorted(hyperparameters)
        print(f"Final features to analyze: {hyperparameters}")
        
    except Exception as e:
        print(f"Error loading data: {e}")

    if group_by_metric:
        df_columns = ['dataset', 'model', 'metric'] + hyperparameters + ['outcome']
    else:
        df_columns = ['dataset'] + hyperparameters + ['outcome']
    
    df = pd.DataFrame(columns=df_columns)

    for index, row in raw_df.iterrows():
        new_row = []
        
        new_row.append(row['DS.Name']) 
        
        if group_by_metric:
            new_row.append(row.get('Model.Name', 'Unknown'))
            new_row.append(row.get('Metric.Name', 'Unknown'))

        for hyperparameter in hyperparameters:
            if hyperparameter in row.index and pd.notna(row[hyperparameter]):
                new_row.append(row[hyperparameter])
            else:
                new_row.append(None)
        
        if outcome_column in row.index and pd.notna(row[outcome_column]):
            if outcome_column=="Time.Duration":
                new_row.append(float(row[outcome_column])/1e9)
            else:
                new_row.append(float(row[outcome_column]))

        else:
            new_row.append(None)
        
        df.loc[index] = new_row

    for index, hyperparameter in enumerate(hyperparameters):
        if index in encode:
            label_encoder = LabelEncoder()
            label_encoder.random_state = RANDOM_SEED 
            df[hyperparameter] = label_encoder.fit_transform(df[hyperparameter])
            print(label_encoder.classes_)

    df = df.dropna()
    print(f"After cleaning: {len(df)} experiments remain")

    df = df.sort_values(['dataset'] + [col for col in sorted(df.columns) if col != 'dataset']).reset_index(drop=True)

    scaler = StandardScaler()

    exclude_cols = ['dataset', 'outcome']
    if group_by_metric:
        exclude_cols.extend(['model', 'metric'])
    
    numeric_cols = [col for col in df.columns if col not in exclude_cols]
    numeric_cols = sorted(numeric_cols)
    print(f"Numeric columns to be normalized: {numeric_cols}")

    # if numeric_cols:
    #     df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    #     print("Features normalized using StandardScaler")
    # else:
    #     print("No numeric columns found to normalize")

    if not numeric_cols:
        print("No numeric columns found to normalize")

    print(f"Processing {len(df)} total experiments")

    group_results = defaultdict(lambda: defaultdict(dict))

    if group_by_metric:
        grouped = df.groupby('metric')
        print(f"Found {len(grouped)} unique metrics")
        
        for metric, group_data in grouped:
            if len(group_data) > 1:
                try:
                    group_key = f"{metric}"

                    analysis_data = group_data.reset_index(drop=True)

                    group_results[group_key]['data'] = analysis_data.copy(deep=True)

                    if numeric_cols:
                        analysis_data[numeric_cols] = scaler.fit_transform(analysis_data[numeric_cols])
                        print("Features normalized using StandardScaler")
                    
                    score = compute_score(analysis_data, hyperparameters, 'outcome')
                    
                    for feature in score.index:
                        effect_value = score.loc[feature, 'outcome']
                        group_results[group_key]['effects'][feature] = round(float(effect_value), 8)
                    
                    group_results[group_key]['effects'] = dict(sorted(group_results[group_key]['effects'].items(), key=lambda x: abs(x[1]), reverse=True))

                    group_results[group_key]['summary'] = f"Effects on {metric} ({len(group_data)} experiments)"

                    print(f"Analyzed {group_key}: {len(analysis_data)} experiments")
                except Exception as e:
                    print(f"Error analyzing {metric}: {e}")
                    continue
        
    else:
        if len(df) > 1: 
            try:
                group_key = outcome_column

                analysis_data = df.drop(columns=['dataset'])

                group_results[group_key]['data'] = analysis_data.copy(deep=True)

                if numeric_cols:
                    analysis_data[numeric_cols] = scaler.fit_transform(analysis_data[numeric_cols])
                    print("Features normalized using StandardScaler")
                
                score = compute_score(analysis_data, hyperparameters, 'outcome')

                for feature in score.index:
                    effect_value = score.loc[feature, 'outcome']
                    group_results[group_key]['effects'][feature] = round(float(effect_value), 8)
                
                group_results[group_key]['effects'] = dict(sorted(group_results[group_key]['effects'].items(), key=lambda x: abs(x[1]), reverse=True))

                group_results[group_key]['summary'] = f"Effects on {group_key} ({len(df)} experiments)"

                print(f"Analyzed {group_key}: {len(analysis_data)} experiments")
            except Exception as e:
                print(f"Error in causal analysis: {e}")

    return group_results, download_dir
