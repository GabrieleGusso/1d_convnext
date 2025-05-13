# by G. Gusso - gabriele.gusso@roma1.infn.it
# Department of Physics - Sapienza University - Rome

# Change PATH with your path

import os
import re
import numpy as np
import sys
import torch

# Input arguments
if len(sys.argv) < 3:
    print("Usage: python mean_std_tests.py <eval_data_path> <output_file>")
    sys.exit(1)

eval_data_path = sys.argv[1]
output_file = sys.argv[2]

# Extract info from header
def extract_header_info(header_line):
    patterns = {
        "batch_size": r"batch_size\s*=\s*(\d+)",
        "data_set": r"data_set\s*=\s*(\S+)",
        "drop_path": r"drop_path\s*=\s*([\d.]+)",
        "smoothing": r"smoothing\s*=\s*([\d.]+)",
        "epochs": r"epochs\s*=\s*(\d+)",
        "lr": r"lr\s*=\s*([\d.eE+-]+)"
    }
    return {k: re.search(p, header_line).group(1) for k, p in patterns.items() if re.search(p, header_line)}

# Main function to process evaluation results
def process_val_history(file_path, ROC_file_path=None):
    with open(file_path, "r") as f:
        lines = f.readlines()

    header_line = lines[0].strip().replace("#", "").strip()
    header_info = extract_header_info(header_line)

    if not header_info:
        print("Could not extract header information.")
        return

    data = []
    tp = tn = fp = fn = None

    for line in lines[1:]:
        if line.strip().startswith("ConfMatr"):
            # Parse confusion matrix values
            parts = line.strip().split()
            if len(parts) == 6:
                _, _, tp, tn, fp, fn = parts
                tp, tn, fp, fn = int(tp), int(tn), int(fp), int(fn)
        else:
            parts = line.strip().split()
            if len(parts) == 4:
                epoch, iteration, loss, acc = int(parts[0]), int(parts[1]), float(parts[2]), float(parts[3])
                data.append((epoch, iteration, loss, acc))

    if not data:
        print("No data found in file.")
        return

    last_epoch = max(d[0] for d in data)
    last_epoch_data = [(it, loss, acc) for (ep, it, loss, acc) in data if ep == last_epoch]

    if last_epoch_data:
        _, losses, accuracies = zip(*last_epoch_data)
        batch_size = float(header_info.get("batch_size", 1))

        mean_loss = np.mean(losses)
        std_loss = np.std(losses, ddof=1) / batch_size
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies, ddof=1) / batch_size

        if ROC_file_path:
            scores = []
            labels = []
            with open(ROC_file_path, "r") as roc_f:
                for line in roc_f:
                    parts = line.strip().split()
                    if len(parts) == 3:
                        _, score, label = parts
                        scores.append(float(score))
                        labels.append(int(label))

            if scores and labels:
                scores_tensor = torch.tensor(scores)
                labels_tensor = torch.tensor(labels)

                # Sort scores in descending order
                sorted_scores, indices = torch.sort(scores_tensor, descending=True)
                sorted_labels = labels_tensor[indices]

                # Compute TPR and FPR at each threshold
                pos_total = (sorted_labels == 1).sum().item()
                neg_total = (sorted_labels == 0).sum().item()

                tps = torch.cumsum(sorted_labels == 1, dim=0).float()
                fps = torch.cumsum(sorted_labels == 0, dim=0).float()

                tpr = tps / pos_total if pos_total > 0 else torch.zeros_like(tps)
                fpr = fps / neg_total if neg_total > 0 else torch.zeros_like(fps)

                # Use trapezoidal rule to compute AUC
                auc = torch.trapz(tpr, fpr).item()
            else:
                auc = float('nan')
        else:
            auc = float('nan')

        print(f"Eval path: {eval_data_path}")
        print(f"Loss: {mean_loss:.6f} ± {std_loss:.6f}")
        print(f"Acc : {mean_acc:.6f} ± {std_acc:.6f}")
        if tp is not None:
            print(f"Confusion Matrix: {tp}, {tn}, {fp}, {fn}")
        if auc is not None:
            print(f"AUC : {auc:.6f}")

        # Append result to output file
        with open(output_file, "a") as out_f:
            if tp is not None:
                if auc is not None:
                    out_f.write(f"{eval_data_path} {mean_loss:.6f} {std_loss:.6f} {mean_acc:.6f} {std_acc:.6f} {tp} {tn} {fp} {fn} {auc:.6f}\n")
                else:
                    out_f.write(f"{eval_data_path} {mean_loss:.6f} {std_loss:.6f} {mean_acc:.6f} {std_acc:.6f} {tp} {tn} {fp} {fn} {auc:.6f}\n")
            else:
                out_f.write(f"{eval_data_path} {mean_loss:.6f} {std_loss:.6f} {mean_acc:.6f} {std_acc:.6f}\n")
    else:
        print("No data found for the last epoch.")

# Find latest history file
history_dir = "/PATH/1d_convnext/results/history/"
latest_var_file = None
latest_ROC_file = None

for fname in sorted(os.listdir(history_dir), reverse=True):
    if fname.endswith("_val_history.txt"):
        latest_var_file = os.path.join(history_dir, fname)
        break

for fname in sorted(os.listdir(history_dir), reverse=True):
    if fname.endswith("_ROC_history.txt"):
        latest_ROC_file = os.path.join(history_dir, fname)
        break

if latest_var_file:
    if latest_ROC_file:
        print("Validation history and ROC files found.")
        process_val_history(latest_var_file, latest_ROC_file)
    else:
        print("Validation history file found.")
        process_val_history(latest_var_file)
else:
    print("No validation history file found.")
