
# Import statements and directory setup
import os
import pandas as pd 
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import warnings
import io
import glob

from sklearn.preprocessing import MinMaxScaler

def piezometer_measurements(series_name):
    # Split series_name handling extra dashes
    parts = series_name.split('-')
    if len(parts) >= 2:
        locatie = parts[0]
        filternummer = '-'.join(parts[1:])
    else:
        warnings.warn(f"Unexpected format for series_name: {series_name}")
        return None

    # Absolute path to raw data
    data_root = Path(r"C:\Users\danielg\PycharmProjects\Taccari_et_al\GroundwaterFlowGNN-main\data\input")
    folder = data_root / "piezometers" / "csv" / "csv"

    # Build glob pattern
    pattern = f"{locatie}{filternummer}*"
    matches = list(folder.glob(pattern))
    #print(f"[DEBUG] Looking for files in {folder} matching {pattern}: found {len(matches)} files")
    if not matches:
        raise FileNotFoundError(f"No files matching {folder/ pattern}")

    df = None
    # Parse each matched file for data blocks
    for filepath in matches:
        lines = filepath.read_text().splitlines()
        start_index = []
        end_index = []
        for i, line in enumerate(lines):
            if line.startswith('LOCATIE,FILTERNUMMER'):
                start_index.append(i)
                # for the first block's end, we find the next blank line (or next header)
                if len(start_index) == 1:
                    continue
                # if it's the second header, mark its start but we still need its end
            elif start_index and len(end_index) < len(start_index):
                # once we have a header, the next empty line or the end of file is its end
                if not line.strip():  # blank line
                    end_index.append(i)
                    # don't break — we need to find both ends
                elif i == len(lines) - 1:
                    end_index.append(i + 1)

        # 2) decide which block to read
        if len(start_index) >= 2 and len(end_index) >= 2:
            # use the *second* block
            s, e = start_index[1], end_index[1]
        elif len(start_index) >= 1 and len(end_index) >= 1:
            # fallback to the *first* block
            s, e = start_index[0], end_index[0]
        else:
            raise ValueError(f"Couldn't find any complete data block in {filename!r}")

        # 3) read that block
        block = io.StringIO('\n'.join(lines[s:e]))
        df = pd.read_csv(block, engine='python', header=0, dtype={'FILTERNUMMER': str})


    if df is None:
        raise FileNotFoundError(f"No valid data block in files matching {folder/ pattern}")

    # Clean and filter
    df.rename(columns={"STAND (cm NAP)": "HEAD"}, inplace=True)
    df['date'] = pd.to_datetime(df['PEIL DATUM TIJD'])
    df['BIJZONDERHEID'] = df['BIJZONDERHEID'].astype(str).str.strip()
    df = df[df['BIJZONDERHEID'] == 'reliable']

    return df


def select_nodes(complete_daily, n_nodes_selection):
    """
    # The `select_nodes` function selects nodes (piezometers) with the least amount of missing data (`NaN` values) from a given DataFrame. 
    
    Parameters:
    - complete_daily: DataFrame containing the data
    - n_nodes_selection: Number of nodes to select
    
    Returns:
    - filtered_df: DataFrame containing only the selected nodes
    """
    
    # Calculate the percentage of NaN values per column for the filtered data
    nan_percentages = complete_daily.isnull().mean()

    # Sort the nodes based on their NaN percentages
    sorted_nodes = nan_percentages.sort_values().index[:n_nodes_selection]

    # Filter the original dataframe to only include the selected nodes
    filtered_df = complete_daily[sorted_nodes]
    
    return filtered_df




def resample_df(df_series, common_start_date, common_end_date, resampling_freq ):
    df_series_2D = df_series.resample(resampling_freq, origin=common_start_date).mean()
    new_index = pd.date_range(start=common_start_date, end=common_end_date, freq=resampling_freq)
    df_series_2D = df_series_2D.reindex(new_index)
    return df_series_2D


def read_and_process_data(file_path):
    """
    Read and process individual data files, converting measurement values and handling missing data.
    """
    data = pd.read_csv(file_path, skiprows=6, delim_whitespace=True, header=None, usecols=[0, 1, 2])
    data.columns = ['Date', 'Measurement', 'Computed_MORIA']
    data['Date'] = pd.to_datetime(data['Date'], format='%Y%m%d%H%M%S')
    data.set_index('Date', inplace=True)
    data['Measurement'] = data['Measurement'].replace(-9999.0000000, np.nan) * 100
    data['Computed_MORIA'] = data['Computed_MORIA'] * 100
    return data


def aggregate_piezo_data(data_path):
    """
    Aggregate data from all piezometer files within a specified directory into a single DataFrame.
    """
    df_piezo_moria = pd.DataFrame()
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith(".txt"):
                file_path = Path(root) / file
                data = read_and_process_data(file_path)
                piezometer_name = file[:-4]
                if 'Computed_MORIA' in data:
                    df_piezo_moria = pd.concat([df_piezo_moria, data['Computed_MORIA'].rename(piezometer_name)], axis=1)
    df_piezo_moria = df_piezo_moria.align(df_piezo_moria, join='outer')[0]
    return df_piezo_moria


def euclidean_distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def print_combined_statistics(df, df_name):
    combined_data = df.values.flatten()
    print(f"{df_name} Combined Statistics:")
    print(f"Mean: {np.mean(combined_data):.2f}")
    print(f"Median: {np.median(combined_data):.2f}")
    print(f"Standard Deviation: {np.std(combined_data):.2f}")
    print(f"Min: {np.min(combined_data):.2f}")
    print(f"Max: {np.max(combined_data):.2f}\n")




