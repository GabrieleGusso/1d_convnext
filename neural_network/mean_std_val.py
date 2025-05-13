# by G. Gusso - gabriele.gusso@roma1.infn.it
# Department of Physics - Sapienza University - Rome

# Change PATH with your path

import os
import re
import numpy as np

# Function to extract key information from the header
def extract_header_info(header_line):
    """
    Extracts batch_size, data_set, drop_path, epochs, and lr from the header line.
    Assumes the header contains these keywords followed by their values.
    """
    header_info = {}

    # Define regex patterns to extract values
    patterns = {
        "batch_size": r"batch_size\s*=\s*(\d+)",
        "data_set": r"data_set\s*=\s*(\S+)",
        "drop_path": r"drop_path\s*=\s*([\d.]+)",
        "smoothing": r"smoothing\s*=\s*([\d.]+)",
        "epochs": r"epochs\s*=\s*(\d+)",
        "lr": r"lr\s*=\s*([\d.eE+-]+)"  # Handles scientific notation (e.g., 1e-3)
    }

    # Extract values from the header
    for key, pattern in patterns.items():
        match = re.search(pattern, header_line)
        if match:
            header_info[key] = match.group(1)
    
    return header_info

# Function to read the file and process the data
def process_val_history(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()

    # Extract and process the header
    header_line = lines[0].strip().replace("#", "").strip()
    header_info = extract_header_info(header_line)

    if not header_info:
        print("Could not extract header information correctly.")
        return

    # Print header information
    print("\nHeader Information:")
    print("=" * 50)
    for key, value in header_info.items():
        print(f"{key.replace('_', ' ').title()}: {value}")
    print("=" * 50)

    # Extract epoch-wise data
    data = []
    for line in lines[1:]:
        parts = line.strip().split()
        if len(parts) == 4:  # Ensure the line contains relevant data
            epoch, iteration, loss, acc = int(parts[0]), int(parts[1]), float(parts[2]), float(parts[3])
            data.append((epoch, iteration, loss, acc))

    # Ensure data is available
    if not data:
        print("No valid data found.")
        return

    # Find the last epoch
    last_epoch = max(epoch for epoch, _, _, _ in data)
    last_epoch_data = [(iteration, loss, acc) for epoch, iteration, loss, acc in data if epoch == last_epoch]

    # Compute mean and std for the last epoch
    if last_epoch_data:
        _, losses, accuracies = zip(*last_epoch_data)

        # Compute loss and accuracy mean/std across all iterations in the last epoch
        mean_loss, std_loss = np.mean(losses), np.std(losses,ddof=1)/float(header_info["batch_size"])
        mean_acc, std_acc = np.mean(accuracies), np.std(accuracies,ddof=1)/float(header_info["batch_size"])

        print(f"\nLast Epoch: {last_epoch+1}")
        print(f"Mean Validation Loss: {mean_loss:.6f} ± {std_loss:.6f}")
        print(f"Mean Validation Accuracy: {mean_acc:.6f} ± {std_acc:.6f}")
    else:
        print("No data found for the last epoch.")

# Find the latest validation history file in the directory
directory = "/PATH/1d_convnext/results/history/" 
latest_file = None

for file in sorted(os.listdir(directory), reverse=True):  # Sort to get the latest file
    if file.endswith("_val_history.txt"):
        latest_file = os.path.join(directory, file)
        break

if latest_file:
    process_val_history(latest_file)
else:
    print("No validation history file found in the directory.")
