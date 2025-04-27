import numpy as np
import pandas as pd
import torch
import random

def euclidean_distance(x1, y1, x2, y2):
    return np.hypot(x2 - x1, y2 - y1)

def load_piezo_metadata(piezo_metadata_path, obs_well_ids):
    """
    Reads well_metadata.csv, filters to only those external_id in obs_well_ids (in order),
    and returns numpy arrays of X, Y, Z plus the count.
    """
    md = pd.read_csv(piezo_metadata_path)
    # filter by external_id (your distances CSV index)
    md = (
        md[md['external_id'].isin(obs_well_ids)]
          .set_index('external_id')
          .reindex(obs_well_ids)
          .reset_index()
    )
    x = md['x_coord'].to_numpy()
    y = md['y_coord'].to_numpy()
    z = md['ground_elev'].to_numpy()
    return x, y, z, len(x)


def load_pump_distances_wide(distance_csv_path, obs_wells, pump_wells):
    # Read the full distance matrix
    mat = pd.read_csv(distance_csv_path, index_col=0)

    # Restrict to exactly the obs & pump wells you need, in the right order:
    #   — obs_wells (200 IDs)
    #   — pump_wells (6 IDs)
    mat = mat.reindex(index=obs_wells, columns=pump_wells)

    # Now melt to long form
    long_df = (
        mat
        .reset_index()
        .melt(id_vars=mat.index.name or 'index',
              var_name='pump_well',
              value_name='distance_m')
        .rename(columns={mat.index.name or 'index': 'obs_well'})
    )
    piezo_idx = {pz: i for i, pz in enumerate(obs_wells)}
    pump_idx = {pm: i for i, pm in enumerate(pump_wells)}

    return long_df, pump_wells, obs_wells, pump_idx, piezo_idx, len(pump_wells)


def generate_complex_adjacency_matrix(
    piezo_x, piezo_y, pump_dist_df, pump_idx, piezo_idx,
    num_piezo, num_pump, percentage=None, n_piezo_connected=3
):
    num_nodes = num_piezo + num_pump
    adj = np.zeros((num_nodes, num_nodes), dtype=float)
    dist = np.full((num_nodes, num_nodes), np.inf, dtype=float)

    # piezo→piezo Euclid
    for i in range(num_piezo):
        for j in range(num_piezo):
            dist[i,j] = euclidean_distance(
                piezo_x[i], piezo_y[i],
                piezo_x[j], piezo_y[j]
            )

    # piezo↔pump from long_df
    for _, row in pump_dist_df.iterrows():
        pz = row['obs_well']
        pm = row['pump_well']
        d  = row['distance_m']
        i = piezo_idx[pz]
        j = num_piezo + pump_idx[pm]
        dist[i,j] = d
        dist[j,i] = d

    # connect nearest piezos
    for i in range(num_piezo):
        nearest = np.argsort(dist[i, :num_piezo])
        chosen = nearest[nearest != i][:n_piezo_connected]
        adj[i, chosen] = 0.1

    # optionally connect piezo→pump
    if percentage is not None:
        cnt = int(np.ceil(num_nodes * (percentage/100)))
        idxs = random.sample(list(range(num_nodes)), cnt)
    else:
        idxs = range(num_nodes)

    for i in idxs:
        if i < num_piezo:
            j_rel = np.argmin(dist[i, num_piezo:num_piezo+num_pump])
            adj[i, num_piezo + j_rel] = 0.2

    return adj + adj.T

def normalize(arr):
    return (arr - arr.min()) / (arr.max() - arr.min())

def create_static_features(pz_x, pz_y, pz_z, num_piezo, num_pump):
    x = normalize(pz_x); y = normalize(pz_y); z = normalize(pz_z)
    x_p = np.zeros(num_pump); y_p = np.zeros(num_pump); z_p = np.zeros(num_pump)
    t_pz = np.ones(num_piezo); t_pm = 2*np.ones(num_pump)
    feats = np.column_stack([
        np.concatenate([x, x_p]),
        np.concatenate([y, y_p]),
        np.concatenate([z, z_p]),
        np.concatenate([t_pz, t_pm])
    ])
    return torch.tensor(feats, dtype=torch.float32)

def main(df_piezo_columns, pump_columns, distance_csv, piezo_metadata_csv, percentage, n_piezo_connected):
    # 1) piezo coords as before
    pz_x, pz_y, pz_z, num_piezo = load_piezo_metadata(piezo_metadata_csv, df_piezo_columns)

    # 2) distances, *now* passing in the same df_piezo_columns and pump_columns
    dist_df, pumps, piezos, pump_idx, piezo_idx, num_pump = \
        load_pump_distances_wide(distance_csv, df_piezo_columns, pump_columns)

    # 3) rest of adjacency build is unchanged
    adj = generate_complex_adjacency_matrix(
        pz_x, pz_y,
        dist_df, pump_idx, piezo_idx,
        num_piezo, num_pump,
        percentage, n_piezo_connected
    )

    static_feats = create_static_features(pz_x, pz_y, pz_z, num_piezo, num_pump)

    return torch.tensor(adj, dtype=torch.float32), static_feats

if __name__ == "__main__":
    obs_list = None   # not used directly, piezo IDs come from your distances file
    adj, feats = main(
        obs_wells=None,
        distance_csv =  "C:\\Users\\danielg\\PycharmProjects\\Taccari_et_al\\GroundwaterFlowGNN-main\\data\\input\\wells\\well_metadata_public.csv",
        piezo_metadata_csv = "C:\\Users\\danielg\\PycharmProjects\\Taccari_et_al\\GroundwaterFlowGNN-main\\data\\preprocessed\\well_metadata.csv",
        percentage=50,
        n_piezo_connected=3
    )
    print("Adjacency matrix:", adj.shape)
    print("Static features:", feats.shape)
