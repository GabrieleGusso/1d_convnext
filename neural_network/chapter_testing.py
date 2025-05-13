import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.ticker import ScalarFormatter

# Change PATH with your paths

data_path = "/PATH/1d_convnext/neural_network/TEST_eval/" # Modify with your path
plot_path = "/PATH/1d_convnext/neural_network/plots/" # Modify with your path

def image_font(size):
	font_path = "/PATH/miniconda3/envs/convnext/fonts/times.ttf"  # Modify with your path
	font_prop = fm.FontProperties(fname=font_path)
	plt.rcParams['font.family'] = font_prop.get_name()
	plt.rcParams.update({
		'font.weight': 'bold',
		'axes.titleweight': 'bold',
		'axes.labelweight': 'bold',
	})
	
	if size == "big":
		plt.rcParams.update({
			'font.size': 18,           # Base font size
			'axes.titlesize': 20,      # Title
			'axes.labelsize': 18,      # Axis labels
			'xtick.labelsize': 16,     # X tick labels
			'ytick.labelsize': 16,     # Y tick labels
			'legend.fontsize': 14,     # Legend
			'figure.titlesize': 22     # Suptitle if used
		})
	elif size == "small":
		plt.rcParams.update({
			'font.size': 16,           # Base font size
			'axes.titlesize': 18,      # Title
			'axes.labelsize': 16,      # Axis labels
			'xtick.labelsize': 14,     # X tick labels
			'ytick.labelsize': 14,     # Y tick labels
			'legend.fontsize': 12,     # Legend
			'figure.titlesize': 20     # Suptitle if used
		})
            
def plot_loss_and_accuracy(
          data_path, 
          plot_path,
          model_name="TrainG_ap", 
          filename="TEST_TrainR-h0.txt", 
          loss_acc="acc",
          color="blue", 
          ax=None,
          hold=False):
    # Construct full path
    file_path = os.path.join(data_path, model_name, filename)

    parameter = filename.split('-')[1][:-4]
    noise = filename.split('-')[0][-1]
    if parameter == "h0":
        par_name = "$h_0$"
        par_search = "h0"
    elif parameter == "P":
        par_name = "$P$"
        par_search = "P"
    elif parameter == "e":
        par_name = "$e$"
        par_search = "ecc"
    if parameter == "ap":
        par_name = "$a_p$"
        par_search = "asini"
    dataset_name = "Test" + noise + "-" + par_name
    
    # Data containers
    var_values = []
    losses = []
    loss_stds = []
    accuracies = []
    acc_stds = []
    AUCs = []

    # Read and parse the file
    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            
            if parameter == "h0":
                match = re.search(rf"{parameter}_(\d+)e(\d+)", parts[0])
                base = int(match.group(1))
                exponent = int(match.group(2))
                var_value = base * 10**(-exponent)
            elif parameter == "P":
                match = re.search(rf"{parameter}_(\d+)", parts[0])
                var_value = int(match.group(1))
            else:
                match = re.search(rf"{par_search}_([0-9.]+)(?:_|$)", parts[0])
                var_value = float(match.group(1))

            var_values.append(var_value)
            
            losses.append(float(parts[1]))
            loss_stds.append(float(parts[2]))
            accuracies.append(float(parts[3]))
            acc_stds.append(float(parts[4]))
            AUCs.append(float(parts[9])*100)

    if loss_acc == "loss":
        norm = np.log(2)
        losses = losses / norm

    image_font("big")
    if ax == None:
        fig, ax = plt.subplots()

    # --- Plot Loss ---
    if loss_acc == "loss":
        ax.plot(var_values, losses, "-", color=color, alpha=0.5, linewidth=1)
        ax.plot(var_values, losses, ".", color=color, markersize=1.5)
        ax.errorbar(var_values, losses, yerr=loss_stds, ecolor=color, linewidth=1, capsize=3, alpha=0.5, fmt='none')
        ax.set_xlabel(par_name)
        ax.set_yscale("log")
        ax.set_ylabel("Loss")
        ax.set_title("Test Losses on " + dataset_name)

    # --- Plot Accuracy ---
    elif loss_acc == "acc":
        ax.plot(var_values, accuracies, "-", color=color, alpha=0.5, linewidth=1)
        ax.plot(var_values, accuracies, ".", color=color, markersize=1.5)
        ax.errorbar(var_values, accuracies, yerr=acc_stds, ecolor=color, linewidth=1, capsize=3, alpha=0.5, fmt='none')
        ax.set_ylim([47.5, 102.5])
        ax.set_xlabel(par_name)
        ax.set_ylabel("Accuracy")
        ax.set_title("Accuracies on " + dataset_name)
    
    # --- Plot Accuracy ---
    elif loss_acc == "AUC":
        ax.plot(var_values, AUCs, "-", color=color, alpha=0.5, linewidth=1)
        ax.plot(var_values, AUCs, ".", color=color, markersize=1.5)
        ax.set_ylim([47.5, 102.5])
        ax.set_xlabel(par_name)
        ax.set_ylabel("AUC")
        ax.set_title("AUC values on " + dataset_name)
    
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 2))
    ax.xaxis.set_major_formatter(formatter)

    ax.grid(True, axis="both")
    savepath = os.path.join(plot_path, f"{filename}_{loss_acc}.png")
    if hold == False:
        plt.draw()
        plt.savefig(savepath, dpi=800, bbox_inches='tight')
        plt.close()
        print("# Figure saved at " + savepath)
    else:
        print("# Holding figure...")

def plot_test_metrics(models, model_names, colors, filenames, gauss_real, AUC):
    for loss_acc in ["acc", "loss"]:
        image_font("big")
        if AUC == True:
            fig, axes = plt.subplots(4, 2, figsize=(12, 20))
        else:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        handles = []
        labels = []

        for j, filename in enumerate(filenames):
            ax = axes[j*2]
            for i in range(len(models)):
                plot_loss_and_accuracy(
                    data_path=data_path,
                    plot_path=plot_path,
                    model_name=models[i],
                    filename=filename,
                    loss_acc=loss_acc,
                    color=colors[i],
                    ax=ax,
                    hold=True
                )
                # Only collect handles once (first subplot only)
                if j == 0:
                    line, = ax.plot([], [], ".", markersize=10, color=colors[i], label=model_names[i])
                    handles.append(line)
                    labels.append(model_names[i])

            ax = axes[j*2+1]
            for i in range(len(models)):
                plot_loss_and_accuracy(
                    data_path=data_path,
                    plot_path=plot_path,
                    model_name=models[i],
                    filename=filename,
                    loss_acc="AUC",
                    color=colors[i],
                    ax=ax,
                    hold=True
                )

        # Add a global legend (bottom center) - reorder for row-wise filling
        ncol = 4
        nrow = len(labels) // ncol
        reordered_handles = []
        reordered_labels = []
        for i in range(ncol):
            for j in range(nrow):
                idx = j * ncol + i
                reordered_handles.append(handles[idx])
                reordered_labels.append(labels[idx])

        fig.legend(reordered_handles, reordered_labels, loc="lower center", ncol=ncol, frameon=True, bbox_to_anchor=(0.5, -0.05))

        plt.tight_layout()
        savepath = os.path.join(plot_path, f"TEST_comparison_{gauss_real}_{loss_acc}.png")
        plt.draw()
        plt.savefig(savepath, dpi=800, bbox_inches='tight')
        plt.close()
        print("# Combined figure saved at " + savepath)


# models        folder names in TEST_eval where each trained model is evaluated on all test datasets
# model_names   names to print in each plot
# colors        colors of each trained model evaluation curve
# filenames     all file names of performance files for each TEST_eval sub-folder
# filenamesG    results for only test datasets with white Gaussian noise
# filenamesR    results for only test datasets with real O3 noise

models = ["TrainR_h0", "TrainR_P", "TrainR_e", "TrainR_ap", "TrainG_h0", "TrainG_P", "TrainG_e", "TrainG_ap"]
model_names = ["TrainR-$h_0$", "TrainR-$P$", "TrainR-$e$", "TrainR-$a_p$", "TrainG-$h_0$", "TrainG-$P$", "TrainG-$e$", "TrainG-$a_p$"]
colors = ["blue", "green", "red", "orange", "cyan", "limegreen", "salmon", "gold"]
filenames = ["TEST_TrainR-h0.txt", "TEST_TrainR-P.txt", "TEST_TrainR-e.txt", "TEST_TrainR-ap.txt", "TEST_TrainG-h0.txt", "TEST_TrainG-P.txt", "TEST_TrainG-e.txt", "TEST_TrainG-ap.txt"]
filenamesG = ["TEST_TrainG-h0.txt", "TEST_TrainG-P.txt", "TEST_TrainG-e.txt", "TEST_TrainG-ap.txt"]
filenamesR = ["TEST_TrainR-h0.txt", "TEST_TrainR-P.txt", "TEST_TrainR-e.txt", "TEST_TrainR-ap.txt"]

plot_test_metrics(models, model_names, colors, filenamesG, "gaussnoise", AUC=True)
plot_test_metrics(models, model_names, colors, filenamesR, "realnoise", AUC=True)
