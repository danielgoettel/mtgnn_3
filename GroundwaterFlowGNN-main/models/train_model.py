import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import json

import sys
sys.path.insert(0, 'C:\\Users\\danielg\\PycharmProjects\\Taccari_et_al\\GroundwaterFlowGNN-main\\data_preprocessing')

from data_preprocessing import process_data 
from data_preprocessing import gnn_data_prep 
from models.mtgnn_lstm import MTGNN_LSTM
from models.mtgnn import MTGNN

from models.lstm_model import LSTMModel
from data_preprocessing.dataset import AutoregressiveTimeSeriesDataset
from utils.training_utils import prepare_combined_input, make_predictions, inverse_transform_with_shape_adjustment, generate_model_filename, save_rmse_values, record_result, analyze_results, get_synthetic

from utils.metrics import calculate_rmse_per_piezometer, calculate_rmse_per_piezometer_moria, print_mean_std
from utils.visualization import plot_sequences, plot_sparsity_pattern, plot_comparison_sequence, plot_comparison_sequence_dual_y, plot_rmse_comparison
import time


from functools import partial


def create_mtgnn_model(num_features, num_nodes, seq_length, model_type, **kwargs):
    """
    Initialize the MTGNN model with the specified parameters.
    """
    # Extract MTGNN specific parameters from kwargs
    mtgnn_params = {key: kwargs[key] for key in kwargs if key in {
        'gcn_true', 'build_adj', 'gcn_depth', 'kernel_set', 'kernel_size',
        'dropout', 'subgraph_size', 'node_dim', 'dilation_exponential',
        'conv_channels', 'residual_channels', 'skip_channels', 'end_channels',
        'in_dim', 'out_dim', 'layers', 'propalpha', 'tanhalpha', 
        'layer_norm_affline', 'xd'
    }}
    mtgnn_params['num_nodes'] = num_nodes
    mtgnn_params['seq_length'] = seq_length + 1  # padding to match the external forces
    mtgnn_params['in_dim'] = 1
    mtgnn_params['out_dim'] = 1
    mtgnn_params['xd'] = num_features  # Assuming 'xd' is the number of static features

    return MTGNN(**mtgnn_params) if model_type=='MTGNN' else MTGNN_LSTM(**mtgnn_params)


def create_lstm_model():
    input_size = 219  
    hidden_size = 150
    output_size = 200  
    external_forces_size = 19
    dense_output_size = 100

    # Correct input size calculation
    return LSTMModel(input_size, hidden_size, output_size, external_forces_size, dense_output_size)



def train(model, optimizer, loss_function, device, num_epochs, train_data, val_data, train_mask, val_mask, df_piezo_columns, num_piezo, static_features, A_tilde, F_w, W, config, model_type):
    
    # Early stopping parameters
    early_stopping_patience = 50
    min_delta = 0.001
    best_loss = float('inf')
            
    # Learning rate scheduler setup
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)
    
    # Create a directory for saving models if it doesn't exist
    os.makedirs("saved_models", exist_ok=True)
    os.makedirs("failed_runs", exist_ok=True)  # Ensure the directory exists
    failed_runs_filepath = "failed_runs/failed_models.txt"

    losses_dict = {"train_losses": {}, "eval_losses": {}}
 
    best_model_filename = None
    

    for future_window in range(1, F_w + 1):  # Gradually increasing the future window
        start_time_window = time.time()  # Start time for the current window
        epoch_times = []  # List to store the duration of each epoch


        losses_dict["train_losses"][f"window_{future_window}"] = []
        losses_dict["eval_losses"][f"window_{future_window}"] = []
    
        # print(f"Training with future window: {future_window}")
        # Adjust the dataset for the current future window
    
        best_loss = float('inf')
        patience_counter = 0
    
        # Load the best model from the previous window if available
        if best_model_filename is not None:
            model.load_state_dict(torch.load(best_model_filename))
            # print(f"Loaded best model from {best_model_filename} for future window: {future_window}")
    
    
        # Model filename for the current future window
        model_filename = generate_model_filename(model_type, future_window,**config)
        # Remove the .pt extension from the model name for the losses
        model_name_for_losses = model_filename.split('/')[-1].replace('.pt', '')


    
        # Check if the model already exists
        if os.path.exists(model_filename):
            model.load_state_dict(torch.load(model_filename))
            # print(f"Loaded model from {model_filename}. Skipping training.")
            continue
    
        # start_time_loading_train_dataset = time.time()

        train_dataset = AutoregressiveTimeSeriesDataset(train_data, W, future_window, train_mask, num_piezo)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        eval_dataset = AutoregressiveTimeSeriesDataset(val_data, W, future_window, val_mask, num_piezo)  
        eval_loader = DataLoader(eval_dataset, batch_size=32, shuffle=False)
        # end_time_loading_train_dataset = time.time()
        # print(f"Loading training and eval dataset took {end_time_loading_train_dataset - start_time_loading_train_dataset:.2f} seconds")
            
        # restart early stopping for every window
        patience_counter = 0
        try:
        
            for epoch in range(num_epochs):
                start_time_epoch = time.time()  # Start time for the current epoch
    
                model.train()
                total_loss = 0
        
                for input_sequence, external_forces_sequence, target_sequence, mask_sequence in train_loader:
                    # Move data to the device
                    input_sequence = input_sequence.to(device)
                    external_forces_sequence = external_forces_sequence.to(device)
                    target_sequence = target_sequence.to(device)
                    mask_sequence = mask_sequence.to(device)
                    
                    optimizer.zero_grad()
        
                    # Initialize autoregressive loop
                    current_input = input_sequence
                    predictions = []
        
                    
                    for t in range(future_window):
                        current_forces = external_forces_sequence[:, t : (W+t+1), :]
                        combined_input = prepare_combined_input(current_input, current_forces)
        
                        if model_type == 'MTGNN' or 'MTGNN_LSTM':
                            output = model(combined_input, A_tilde.to(device), FE=static_features.to(device)) if not config['build_adj'] else model(combined_input, FE=static_features.to(device))
                        elif model_type == 'LSTM' or 'tCNN':
                            output = model(combined_input, current_forces)
                        else:
                            raise ValueError("Invalid model type. Choose 'MTGNN' or 'LSTM'.")
        
                        output = output[:, :, :num_piezo, 0] if model_type == 'MTGNN' or 'MTGNN_LSTM' else output
                        predictions.append(output)
                        next_input = output
                        current_input = torch.cat((current_input[:, 1:, :], next_input), dim=1)
        
                    # Multi-Step loss
                    predictions = torch.cat(predictions, dim=1)
                    
                    predictions_masked = predictions * mask_sequence
                    target_masked = target_sequence * mask_sequence
                    loss = loss_function(predictions_masked, target_masked)
        
                    # loss = loss_function(predictions, target_sequence)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    del input_sequence, external_forces_sequence, target_sequence, mask_sequence
                    torch.cuda.empty_cache()  # Be cautious with frequent use
        
                train_loss = total_loss / len(train_loader)
        
                model.eval()  # Set the model to evaluation mode
                total_eval_loss = 0
        
                with torch.no_grad():  # Disable gradient computations
                    for input_sequence, external_forces_sequence, target_sequence, mask_sequence in eval_loader:
                        # Move data to the device
                        input_sequence = input_sequence.to(device)
                        external_forces_sequence = external_forces_sequence.to(device)
                        target_sequence = target_sequence.to(device)
                        mask_sequence = mask_sequence.to(device)
        
                        current_input = input_sequence
                        predictions = []
                        for t in range(future_window):  
                            current_forces = external_forces_sequence[:, t : (W+t + 1), :]
                            combined_input = prepare_combined_input(current_input, current_forces)
        
                            if model_type == 'MTGNN' or 'MTGNN_LSTM':
                                output = model(combined_input, A_tilde.to(device), FE=static_features.to(device)) if not config['build_adj'] else model(combined_input, FE=static_features.to(device))
                            elif model_type == 'LSTM' or 'tCNN':
        
                                output = model(combined_input, current_forces)
                            else:
                                raise ValueError("Invalid model type. Choose 'MTGNN' or 'LSTM'.")
        
                            output = output[:, :, :num_piezo, 0] if model_type == 'MTGNN' or 'MTGNN_LSTM' else output
                            predictions.append(output)
                            
                            # Update the input sequence for the next prediction
                            next_input = output 
                            current_input = torch.cat((current_input[:, 1:, :], next_input), dim=1)
        
                        predictions = torch.cat(predictions, dim=1)
        
                        predictions_masked = predictions * mask_sequence
                        target_masked = target_sequence * mask_sequence
                        loss = loss_function(predictions_masked, target_masked)
                        # loss = loss_function(predictions, target_sequence)
                        total_eval_loss += loss.item()
        
                eval_loss = total_eval_loss / len(eval_loader)
    
                losses_dict["train_losses"][f"window_{future_window}"].append({"epoch": epoch + 1, "loss": train_loss})
                losses_dict["eval_losses"][f"window_{future_window}"].append({"epoch": epoch + 1, "loss": eval_loss})
                
                # Print losses every 10 epochs
                if epoch % 10 == 9:
                    print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.2e}, Eval Loss: {eval_loss:.2e}")
        
                # Adaptive Learning Rate
                scheduler.step(eval_loss)
        
                # Early Stopping Check
                if eval_loss + min_delta < best_loss:
                    best_loss = eval_loss
                    patience_counter = 0
                    # Save the best model
                    torch.save(model.state_dict(), model_filename)
                    #print(f"Saved best model to {model_filename}")
                else:
                    patience_counter += 1
        
                if patience_counter >= early_stopping_patience:
                    print("Early stopping triggered.")
                    break
    
                end_time_epoch = time.time()  # End time for the current epoch
                epoch_times.append(end_time_epoch - start_time_epoch)  # Store the duration of the epoch

                # print(f"Epoch {epoch + 1} completed in {end_time_epoch - start_time_epoch:.2f} seconds")
            
            # # After completing training for the current window
            end_time_window = time.time()
            total_time_window = end_time_window - start_time_window  # Total time for the window
            average_epoch_time = sum(epoch_times) / len(epoch_times) if epoch_times else 0  # Calculate the average time per epoch
            print(f"Average epoch time for future window {future_window}: {average_epoch_time:.2f} seconds")

            # print(f"Training for window {future_window} completed in {end_time_window - start_time_window:.2f} seconds")
                
        
            # Now you can use model_name_for_losses for naming your loss files
            loss_filename = f'losses_{model_name_for_losses}.json'
            os.makedirs("training_results", exist_ok=True)
    
            loss_filename_filepath = os.path.join("training_results", loss_filename)
            with open(loss_filename_filepath, 'w') as f:
                json.dump(losses_dict, f, indent=4)
                
                
            # At the end of training for each future window, update the best model filename
            best_model_filename = model_filename if os.path.exists(model_filename) else None
                
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print(f"Out of memory when processing {model_filename}.")
                with open(failed_runs_filepath, "a") as file:
                    file.write(f"{model_filename}\\n")
                return None  # Exit the training function early
            else:
                raise  # Re-raise the exception if it's not a memory error


def define_base_configuration():
    return {
        'synthetic_data': False,
        'percentage': None,
        'n_piezo_connected': 3,
        'W': 5,
        'gcn_true': True,
        'build_adj': False,
        'gcn_depth': 4,
        'kernel_set': [1, 2],
        'kernel_size': 2,
        'dropout': 0.5,
        'subgraph_size': 20,
        'node_dim': 10,
        'dilation_exponential': 2,
        'conv_channels': 64,
        'residual_channels': 64,
        'skip_channels': 64,
        'end_channels': 128,
        'in_dim': 1,  # Not varied
        'out_dim': 1,  # Not varied
        'layers': 4,
        'propalpha': 0.07,
        'tanhalpha': 0.2,
        'layer_norm_affline': True
    }

# Define variations for each parameter (exclude 'in_dim' and 'out_dim')
parameter_variations = {
    'synthetic_data': [False],
    'percentage': [50],
    'n_piezo_connected': [1, 4],
    'build_adj': [True],

    'W': [3, 7],
    
    'gcn_true': [False],
    'gcn_depth': [2],
    # 'kernel_set': [[1, 3], [1, 2]],
    # 'kernel_size': [3, 4],
    'dropout': [0.3, 0.7],
    'subgraph_size': [10, 25],
    'node_dim': [8, 12],
    'dilation_exponential': [1, 3],
    'conv_channels': [32, 128],
    'residual_channels': [32, 128],
    'skip_channels': [32, 128],
    'end_channels': [64, 256],
    'layers': [3, 5],
    'propalpha': [0.05, 0.1],
    'tanhalpha': [0.1, 0.3],
    'layer_norm_affline': [False]
}

def generate_configurations():
    base_config = define_base_configuration()
    configs = [base_config]  # Start with the base configuration
    
    # Iterate over each parameter and its variations
    for param, variations in parameter_variations.items():
        for variation in variations:
            new_config = base_config.copy()
            new_config[param] = variation
            configs.append(new_config)
    
    return configs


def main(run_all=True):
    configs = generate_configurations()
    base_config = configs[0]  # The first configuration is the base configuration

    if run_all:
        total_runs = len(configs)  # Total number of configurations to run including sensitivity analysis
    else:
        total_runs = 1  # Only run the base configuration

    for i, config in enumerate(configs[:total_runs], start=1):  # Limit the configs based on run_all flag
        differing_params = {k: v for k, v in config.items() if base_config.get(k) != v}
        differing_params_str = ', '.join([f'{key}: {value}' for key, value in differing_params.items()])
        
        if differing_params:
            print(f"Running configuration {i} of {total_runs} with variation: {differing_params_str}")
        else:
            print(f"Running base configuration {i} of {total_runs}")
        
        test_rmse_mean, test_rmse_std = run_training_and_evaluation(config)
        
        # Record the result
        record_result(config, test_rmse_mean, test_rmse_std)
    
    if run_all:
        # After all configurations have been tested, analyze the results
        analyze_results()


def run_training_and_evaluation(config):
  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device.")

    # Assuming process_data.main() prepares and returns the necessary datasets and GNN data
    train_data, val_data, test_data, train_mask, val_mask, test_mask, df_piezo_columns, pump_columns, locations_no_missing, scaler = process_data.main(config['synthetic_data'])
    A_tilde, static_features = gnn_data_prep.main(df_piezo_columns, pump_columns, locations_no_missing, config['percentage'] , config['n_piezo_connected'] )
    # plot_sparsity_pattern(A_tilde, markersize=10)

    num_piezo = len(df_piezo_columns)
    num_features = static_features.shape[1]  # Assuming static_features is a tensor
    seq_length = config['W']  # sequence length
    num_nodes = A_tilde.shape[0]
    
    # Specify the sequence length (W) and future window size
    W = config['W']

    # for F_w in range(1,12): 
    F_w = 3 # maximum future window size

    model_type = 'MTGNN'
    if model_type=='MTGNN' or 'MTGNN_LSTM':
        model = create_mtgnn_model(
            num_features=num_features, 
            num_nodes=num_nodes, 
            seq_length=seq_length,
            model_type=model_type,
            **config  # Unpacks and passes the configuration dictionary
        ).to(device)
    else: 
        model = create_lstm_model().to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_function = nn.MSELoss()

    train(model, optimizer, loss_function, device, num_epochs=200, train_data=train_data, val_data=val_data, train_mask=train_mask, val_mask=val_mask, df_piezo_columns=df_piezo_columns, num_piezo=num_piezo, static_features=static_features, A_tilde=A_tilde, F_w=F_w, W=W, config = config, model_type = model_type)


    # Selecting first samples from training and testing datasets
    test_sample = AutoregressiveTimeSeriesDataset(test_data, input_window=W, max_future_window=100, missing_data_mask = test_mask, num_piezo = num_piezo)[1]
    test_input, test_predicted_model, test_target = make_predictions(model, test_sample, device, 100, W, A_tilde, static_features, num_piezo, modeltype=model_type)
        
    # Transform predictions back to original scale

    test_predicted_model_ = inverse_transform_with_shape_adjustment(test_predicted_model.numpy(), scaler, num_piezo)
    test_input_ = inverse_transform_with_shape_adjustment(test_input.numpy(), scaler, num_piezo)
    test_target_ = inverse_transform_with_shape_adjustment(test_target.numpy(), scaler, num_piezo)

    test_rmse = calculate_rmse_per_piezometer(test_predicted_model_, test_target_, num_piezo)

    test_rmse_mean, test_rmse_std = print_mean_std(test_rmse, "Test RMSE")

    save_rmse_values(test_rmse, model_type, F_w,**config)

    # Plotting Model 1 Predictions
    _, _, _, mask_seq_test = test_sample
    start_date_test = test_data.index[0] 
    # plot_sequences(test_input_, test_predicted_model_, test_target_, df_piezo_columns, 'Model Evaluation', start_date_test, model_labels=('Prediction', '', ''), mask = mask_seq_test)
    color_dict_seq = plot_comparison_sequence(test_input_, test_predicted_model_, test_target_, start_date_test, df_piezo_columns, mask=mask_seq_test, selected_nodes=None)
    
    color_dict_dual = plot_comparison_sequence_dual_y(test_input_, test_predicted_model_, test_target_, start_date_test, mask_seq_test, test_rmse, df_piezo_columns )
    combined_color_dict = {**color_dict_seq, **color_dict_dual}

    df_moria, mask_moria = get_synthetic(df_piezo_columns)
    test_rmse_moria = calculate_rmse_per_piezometer_moria(df_moria, test_target_, df_piezo_columns)
    test_rmse_dict = dict(zip(df_piezo_columns, test_rmse))
    plot_rmse_comparison(test_rmse_moria, test_rmse_dict, 'MODFLOW', 'ST-GNN', combined_color_dict)

    return test_rmse_mean, test_rmse_std

if __name__ == "__main__":
    main(run_all=False) # for running only the base configuration

