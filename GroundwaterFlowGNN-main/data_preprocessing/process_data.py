# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 14:35:06 2024
Edited by DMG on 4/25/2025
@author: cnmlt
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
import glob
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pickle
import hashlib
from pathlib import Path
import os
import pickle

from data_preprocessing.preprocessing import (
    piezometer_measurements,
    resample_df,
    select_nodes,
    read_and_process_data, 
    aggregate_piezo_data )


from utils.visualization import plot_two_random_columns 


DATA_ROOT = Path(r"C:\Users\danielg\PycharmProjects\Taccari_et_al\GroundwaterFlowGNN-main\data\input")


def setup_environment():
    """
    Determine and return the raw-data and preprocessed-data roots,
    creating the preprocessed folder if it doesn’t exist, and
    cd’ing into the GroundwaterFlowGNN-main folder.
    """
    # 1) project_root is GroundwaterFlowGNN-main
    project_root = Path(__file__).parent.parent.resolve()

    # 2) cd into it so all relative paths (like 'data/input') resolve under that
    os.chdir(project_root)

    # 3) define your raw and preprocessed paths
    raw_input_path    = project_root / "data" / "input"
    preprocessed_path = project_root / "data" / "preprocessed"

    # 4) make sure the output folder exists
    preprocessed_path.mkdir(parents=True, exist_ok=True)

    return raw_input_path, preprocessed_path


def load_and_filter_series(aquifer=None):
    """
    Load piezometer series metadata and filter based on criteria.
    """
    df = pd.read_csv('C:\\Users\\danielg\\PycharmProjects\\Taccari_et_al\\GroundwaterFlowGNN-main\\data\\input\\piezometers\\oseries_calibration_settings.csv')
    if aquifer:
        return df[df['aquifer'] == aquifer]['name']
    else:
        return df['name']

def process_series(series_names, start_year=2004, resampling_freq='W'):
    """
    Process each series in the list of series names.
    """
    dfs = []
    for series_name in series_names:
        try:
            df_series = piezometer_measurements(series_name)
            if df_series is not None:
                df_series = df_series.groupby('date').agg({'HEAD': 'mean'})
                df_series = df_series.resample(resampling_freq).mean()
                dfs.append(df_series.rename(columns={'HEAD': series_name}))
        except FileNotFoundError as e:
            print(f"Error processing series {series_name}")

    complete_daily = pd.concat(dfs, axis=1)
    complete_daily = complete_daily[complete_daily.index.year >= start_year]
    return complete_daily


def save_data_with_config_hash(data, config, data_filepath, config_hash_filepath):
    """
    Save data with configuration hash to check for changes.

    Parameters:
    - data: The data to be saved.
    - config: The configuration settings used for processing the data.
    - data_filepath: The file path for saving the processed data.
    - config_hash_filepath: The file path for saving the configuration hash.
    """
    # Serialize data and save
    with open(data_filepath, 'wb') as f:
        pickle.dump(data, f)
    
    # Serialize config hash and save
    config_hash = hashlib.sha256(pickle.dumps(config)).hexdigest()
    with open(config_hash_filepath, 'wb') as f:
        pickle.dump(config_hash, f)


def load_data_if_config_unchanged(config, base_data_path):
    """Load data if configuration hasn't changed."""
    data_type_prefix = "_synthetic" if config['synthetic_data'] else ""
    
    processed_data_filepath = base_data_path / f'processed_data{data_type_prefix}.pkl'
    config_hash_filepath = base_data_path / 'config_hash{data_type_prefix}.pkl'
    
    # Check if config hash file exists and compare with current config
    try:
        with open(config_hash_filepath, 'rb') as f:
            saved_config_hash = pickle.load(f)
        current_config_hash = hashlib.sha256(pickle.dumps(config)).hexdigest()
        
        if saved_config_hash == current_config_hash:
            with open(processed_data_filepath, 'rb') as f:
                print("Loading data from saved file.")
                data = pickle.load(f)
                return data  # Returns the tuple of data (complete_daily, df_piezo, missing_data_mask)
    except FileNotFoundError:
        print("Config hash or processed data file not found. Processing data.")
    
    return None


def save_data(data, data_filepath):
    """
    Save data to a specified file path.

    Parameters:
    - data: The data to be saved.
    - data_filepath: The file path for saving the processed data.
    """
    # Serialize data and save
    with open(data_filepath, 'wb') as f:
        pickle.dump(data, f)

    print("Data saved successfully.")

def load_data(base_data_path, config):
    """
    Load data from a specified file path.
    
    Parameters:
    - base_data_path: The base directory where data files are stored.
    - config: The configuration settings used for determining the data file path.
    
    Returns:
    - Loaded data if the file exists, otherwise None.
    """
    data_type_prefix = "_synthetic" if config['synthetic_data'] else ""
    processed_data_filepath = base_data_path / f'processed_data{data_type_prefix}.pkl'
    
    try:
        with open(processed_data_filepath, 'rb') as f:
            print("Loading data from saved file.")
            data = pickle.load(f)
            return data
    except FileNotFoundError:
        print("Processed data file not found. Processing data.")
    
    return None



def save_column_names(columns, base_data_path, data_type_suffix=''):
    """
    Save column names to a text file, appending a suffix to differentiate between real and synthetic data.

    Parameters:
    - columns: Iterable containing the column names.
    - base_data_path: Path object pointing to the base data directory.
    - data_type_suffix: String indicating the data type ("_synthetic" for synthetic data, "_real" for real data, otherwise an empty string).
    """
    column_names_filepath = base_data_path / f'column_names{data_type_suffix}.txt'
    
    with open(column_names_filepath, 'w') as file:
        for column in columns:
            file.write(column + '\n')


def load_column_names(base_data_path, data_type_suffix=''):
    """
    Load column names from a text file if it exists, appending a suffix to differentiate between real and synthetic data.

    Parameters:
    - base_data_path: Path object pointing to the base data directory.
    - data_type_suffix: String indicating the data type ("_synthetic" for synthetic data, "_real" for real data, otherwise an empty string).

    Returns:
    - A list of column names if the file exists and is loaded successfully, otherwise None.
    """
    column_names_filepath = base_data_path / f'column_names{data_type_suffix}.txt'
    
    # Check if the column names file exists
    if column_names_filepath.exists():
        with open(column_names_filepath, 'r') as file:
            # Read column names into a list, stripping newline characters
            columns = [line.strip() for line in file.readlines()]
        return columns
    else:
        # Return None if the file does not exist
        return None


def fill_and_select_data(df, n_nodes_selection=200):
    """
    Fill missing data and select nodes based on selection criteria.
    """
    
    df =  select_nodes(df, n_nodes_selection)
    
    missing_data_mask = ~df.isna()

    df = df.interpolate(method='linear', order=3).bfill(limit=None)


    return df, missing_data_mask



def process_synthetic_data(synthetic_data_path, real_data_path, series_names, config):
    """
    Incorporate reading, processing, aggregating, deduplicating, and resampling of synthetic data.
    """
    df_piezo_moria = aggregate_piezo_data(synthetic_data_path)
    print(f"Unique columns after deduplication: {len(df_piezo_moria.columns)}")

    filtered_columns = [col for col in series_names if col in df_piezo_moria.columns]
    print(f"Filtered columns count: {len(filtered_columns)}")

    additional_needed = config['n_nodes_selection'] - len(filtered_columns)
    if additional_needed > 0:
        # check columns that are present for both in the metadata and df_piezo_moria, excluding already filtered columns

        piezo_metadata = pd.read_csv(real_data_path/ "piezometers/oseries_metadata_and_selection.csv")  
        common_columns = piezo_metadata[piezo_metadata['name'].isin(df_piezo_moria.columns)]
        potential_additional_columns = common_columns[common_columns['name'].isin(filtered_columns) == False]['name'].tolist()
        # Select randomly without repetition
        selected_randomly = np.random.choice(potential_additional_columns, size=min(additional_needed, len(potential_additional_columns)), replace=False)
        selected_columns = filtered_columns + list(selected_randomly)
    else:
        selected_columns = filtered_columns[:config['n_nodes_selection']]

    df_piezo_moria_selected = df_piezo_moria[selected_columns]
    df_piezo_moria_unique_cols = df_piezo_moria_selected.loc[:, ~df_piezo_moria_selected.columns.duplicated(keep='first')]

    common_start_date = pd.Timestamp(config.get('common_start_date', '2008-01-01'))
    df_piezo_moria_unique_cols.index = pd.to_datetime(df_piezo_moria_unique_cols.index)
    resampled_df = df_piezo_moria_unique_cols.resample(config['resampling_freq'], origin=common_start_date).mean()

    missing_data_mask = ~resampled_df.isna()
    return resampled_df, missing_data_mask


def split_and_normalize_data(df_piezo, missing_data_mask, external_data, config):
    """
    Splits the dataset into training, validation, and test sets, normalizes them, and also splits the missing data mask.

    Parameters:
    - df_piezo: DataFrame containing piezometer data.
    - missing_data_mask: DataFrame indicating the presence of missing data.
    - external_data: A tuple containing DataFrames of external data (pumping wells, precipitation, evaporation, river).
    - config: A dictionary with configuration settings such as split ratios.

    Returns:
    - A tuple of normalized DataFrames: (train_data, val_data, test_data, train_mask, val_mask, test_mask).
    """
    # Combine piezometer data with external data for processing
    combined_data = pd.concat([df_piezo] + list(external_data), axis=1)

    test_val_size = 0.2
    # Split data into train, validation, and test sets initially
    train_data, temp_data = train_test_split(combined_data, test_size=test_val_size, random_state=42, shuffle=False)
    # val_data, test_data = train_test_split(temp_data, test_size=test_val_split, random_state=42, shuffle=False)
    val_data = temp_data
    test_data = temp_data
    
    train_mask, temp_mask = train_test_split(missing_data_mask, test_size=test_val_size, random_state=42, shuffle=False)
    # val_mask, test_mask = train_test_split(temp_mask, test_size=test_val_split, random_state=42, shuffle=False)
    val_mask = temp_mask
    test_mask = temp_mask
    # Normalize the datasets
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_data = pd.DataFrame(scaler.fit_transform(train_data), index=train_data.index, columns=train_data.columns)
    val_data = pd.DataFrame(scaler.transform(val_data), index=val_data.index, columns=val_data.columns)
    test_data = pd.DataFrame(scaler.transform(test_data), index=test_data.index, columns=test_data.columns)

    # Return the normalized datasets along with their corresponding masks
    return train_data, val_data, test_data, train_mask, val_mask, test_mask, scaler


def define_configuration(synthetic_data):
    return {
        'start_year': 2004,
        'resampling_freq': 'W',
        'n_nodes_selection': 200,
        'aquifer': None,
        'synthetic_data': synthetic_data,  
        'plotting': True, 

    }



def load_external_data(data_path, common_start_date, common_end_date, resampling_freq):
    """
    Load and preprocess external data sources including pumping wells data.

    Parameters:
    - data_path: Path to the directory containing external data files.
    - common_start_date: The start date for filtering and resampling the data.
    - common_end_date: The end date for filtering and resampling the data.
    - resampling_freq: Frequency for resampling the time series data.

    Returns:
    - DataFrame: df_pumping_wells
    """
    cache_path = data_path / "pumping_wells_processed.pkl"

    if cache_path.exists():
        print("✅ Loading cached pumping wells data...")
        df_pumping_wells = pd.read_pickle(cache_path)
    else:
        print("⚙️ Processing pumping wells data...")
        pumping_file_1 = data_path / "wells" / "wells_data_1.csv"
        pumping_file_2 = data_path / "wells" / "wells_data_2.csv"

        # Load pumping wells data
        df_pump1 = pd.read_csv(pumping_file_1, header=0, parse_dates=['Datum'], index_col='Datum', na_values ='-')
        df_pump2 = pd.read_csv(pumping_file_2, header=0, parse_dates=['Datum'], index_col='Datum', na_values ='-')

        # Convert df_pump1 from m3/month to m3/day
        days_in_month = df_pump1.index.days_in_month
        df_pump1 = df_pump1.div(days_in_month, axis=0)

        # Concatenate dataframes
        df_pumping_wells = pd.concat([df_pump1, df_pump2])

        # Sort by date index after concatenation
        df_pumping_wells.sort_index(inplace=True)

        # Resample dataframe
        df_pumping_wells = resample_df(df_pumping_wells, common_start_date, common_end_date, resampling_freq).fillna(0)

        # Cache the processed data
        df_pumping_wells.to_pickle(cache_path)
        print("💾 Pumping wells data cached.")

    return df_pumping_wells

def main():
    config = define_configuration(False)

    data_path, base_data_path = setup_environment()

    processed_data_filepath = base_data_path / 'processed_data.pkl'

    # Attempt to load cached df_piezo and mask
    if processed_data_filepath.exists():
        print("✅ Loading cached piezometer data...")
        df_piezo, missing_data_mask = pd.read_pickle(processed_data_filepath)
    else:
        print("⚙️ Processing new piezometer data...")
        series_names = load_and_filter_series(config['aquifer'])
        complete_daily = process_series(series_names, config['start_year'], config['resampling_freq'])
        df_piezo, missing_data_mask = fill_and_select_data(complete_daily, config['n_nodes_selection'])
        save_column_names(df_piezo.columns, base_data_path, '_real')
        pd.to_pickle((df_piezo, missing_data_mask), processed_data_filepath)
        print("💾 Piezometer data cached.")

    common_start_date = pd.Timestamp('2004-01-01')
    common_end_date = df_piezo.index[-1]

    # Load pumping wells data (already cached in load_external_data)
    df_pumping_wells = load_external_data(data_path, common_start_date, common_end_date, config['resampling_freq'])

    # Prepare columns
    df_piezo_columns = df_piezo.columns.tolist()
    pump_columns = df_pumping_wells.columns.tolist()

    # Split and normalize
    train_data, val_data, test_data, train_mask, val_mask, test_mask, scaler = split_and_normalize_data(
        df_piezo, missing_data_mask, (df_pumping_wells,), config
    )

    return train_data, val_data, test_data, train_mask, val_mask, test_mask, df_piezo_columns, pump_columns, scaler


if __name__ == "__main__":
    main()


