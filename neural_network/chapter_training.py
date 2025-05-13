# KARPAHY TESTS

import os, glob, re
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.ticker import ScalarFormatter
from models.convnext import convnext_tiny, convnext_tiny

# Change PATH with your paths

data_path = "/PATH/1d_convnext/results/history/"
plot_path = "/PATH/1d_convnext/neural_network/plots/"

# PLOT FUNCTIONS
class Parameters:
    def __init__(self, **entries):
        self.__dict__.update(entries)


def len_tain_val(dir_path):
    signal_path = dir_path + "/signal"
    signal_items = os.listdir(signal_path)
    signal_len = len(
        [f for f in signal_items if os.path.isfile(os.path.join(signal_path, f))]
    )
    noise_path = dir_path + "/noise"
    noise_items = os.listdir(noise_path)
    noise_len = len(
        [f for f in noise_items if os.path.isfile(os.path.join(noise_path, f))]
    )
    return signal_len, noise_len


def parse_namespace_string(namespace_str):
    # remove prefix and closing parenthesis
    if namespace_str.startswith("# Namespace("):
        cleaned_str = namespace_str[len("# Namespace(") : -2]

    # clean the string
    cleaned_str = cleaned_str.replace("=", ":")
    cleaned_str = re.sub(r"(\w+):", r"'\1':", cleaned_str)
    cleaned_str = re.sub(r"\bFalse\b", "False", cleaned_str)
    cleaned_str = re.sub(r"\bTrue\b", "True", cleaned_str)
    cleaned_str = re.sub(r"\bNone\b", "None", cleaned_str)
    cleaned_str = re.sub(r"''env'://'", "'env://'", cleaned_str)
    cleaned_str = re.sub(r"(?<=\w)(\s)(?=\w+[:])", ", ", cleaned_str)

    # create the dictionary and the string
    params_dict = eval(f"{{{cleaned_str}}}")
    parameters = Parameters(**params_dict)

    # get train and val sizes
    train_signal_len, train_noise_len = len_tain_val(parameters.data_path + "/train")
    val_signal_len, val_noise_len = len_tain_val(parameters.data_path + "/val")
    parameters.train_len = train_signal_len + train_noise_len
    parameters.val_len = val_signal_len + val_noise_len
    parameters.train_val_ratio = parameters.train_len / (
        parameters.train_len + parameters.val_len
    )

    return parameters


def train_val_file(in_file: str, train_val: str):
    if "train" in train_val or train_val == "val" or train_val == "iter":
        out_file = in_file + train_val + "*"
    else:
        TypeError('Enter a valid name: "all", "train", "val" or "iter"')
    return out_file


def date_file(date, train_val):
    if date == None:
        file = train_val_file("*", train_val)
    else:
        file = train_val_file(str(date) + "_", train_val)
    return file


def get_latest_file(date, train_val):
    file = date_file(date, train_val)
    list_of_files = glob.glob(data_path + file)
    if list_of_files:
        latest_file = max(list_of_files, key=os.path.getctime)
    else:
        latest_file = None
    return latest_file

def image_font(size):
	font_path = "/nfsroot/home1/homedirs/gusso/miniconda3/envs/convnext/fonts/times.ttf"  # Modify with the actual path
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
		
def plot_loss_acc(x, y, train_val, loss_acc, plot_path, name, parameters, hold = False, ax = None, color = "blue", print_errorbar = True, font_size = "small"):
    savepath = plot_path + name + loss_acc + ".png"
    if loss_acc == "loss":
        norm = np.log(2)
        y = y / norm

    if ax == None:
        fig, ax = plt.subplots()
    image_font(font_size)

    suptitle = (
        loss_acc + " of " + train_val + " set vs epochs (" + parameters.data_set + ")"
    )
    plt.suptitle(suptitle)

    title = (
        "train="
        + str(parameters.train_len)
        + " val="
        + str(parameters.val_len)
        + " lr="
        + str(parameters.lr)
        + " batch_size="
        + str(parameters.batch_size)
    )
    plt.title(title, fontsize=10)

    if train_val == "train":
        ax.plot(x, y, "-", color=color, alpha=0.5, linewidth=1)
        ax.plot(x, y, ".", color=color, markersize=1.5)
    else:
        max_epoch = round(max(x))+1
        mu = []
        std = []
        # print(max_epoch)
        for i in range(max_epoch):
            indices = np.array(
                [idx for idx, val in enumerate(x) if (val >= i and val < i + 1)]
            )
            if len(indices) > 0:  # ensure indices is not empty
                mu.append(np.mean([y[idx] for idx in indices]))
                std.append(np.std([y[idx] for idx in indices]))
            else:
                mu.append(0)  # placeholder values
                std.append(0)
        ax.plot(np.arange(max_epoch), mu, "-", color=color, alpha=0.5, linewidth=1)
        ax.plot(np.arange(max_epoch), mu, ".", color=color, markersize=1.5)
        # print(mu)
        # print(std)
        if print_errorbar == True:
            ax.errorbar(
                np.arange(max_epoch),
                mu,
                yerr=std/np.sqrt(parameters.batch_size),
                ecolor=color,
                linewidth=1,
                capsize=3,
				alpha=0.5,
                fmt='none',
            )

    if loss_acc == "loss":
        plt.yscale("log")
    #     plt.ylim(0.1, 2)
    # else:
    #     plt.ylim(40, 110)

    ax.grid(True, axis="both")
    plt.xlabel("Epochs")
    plt.ylabel(loss_acc)
    plt.rcParams["font.family"] = "Times New Roman"
    if hold == False:
        plt.draw()
        plt.savefig(savepath)
        plt.close()
        print("# Figure saved at " + savepath)
    else:
        print("# Holding figure...")

    return True

# PRINT MEAN STD FUNCTIONS
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

# verify loss @ init
def loss_at_init():
	model = convnext_tiny(in_chans=1, num_classes=2).to("cuda")
	input_tensor = torch.zeros(1, 1, 3000).to("cuda") # dummy input
	logits = model(input_tensor)
	target = torch.tensor([0], device="cuda")
	criterion = nn.CrossEntropyLoss()
	loss = criterion(logits, target)
	print("loss @ init: {}".format(loss.item()))

# input-indepent baseline
# put "return sample*0.0" as output of matlab_loader() in dataloader.py

# overfit a batch of 2 samples, use:
# --drop_path 0 \
# --smoothing 0 \
# --batch_size 2 \
# --epochs 1000 \
# --data_set "THESIS_realnoise_overfit2samples_f_70_h0_2e24_P_19_ecc_0_asini_1_cosi_0" \
# --disable_eval True \

# input-indipendent baseline
def plot_zero_input(data_path, plot_path, date_zero_input, date_real_input):
	hold = True

	for train_val in ["train", "val"]:
		for loss_acc in ["loss", "acc"]:
			fig, ax = plt.subplots()
			for i, date in enumerate([date_real_input, date_zero_input]):
				if i == 0:
					color = "blue"
				else:
					color = "red"

				latest_file = get_latest_file(date, train_val)

				if latest_file:
					with open(latest_file) as file:
						namespace_str = file.readline()
						parameters = parse_namespace_string(namespace_str)

						lines = file.readlines()
						epochs = np.array([line.split()[0] for line in lines[2:]], dtype=int)
						iterations = np.array([line.split()[1] for line in lines[2:]], dtype=int)
						train_loss = np.array([line.split()[2] for line in lines[2:]], dtype=float)
						train_acc = np.array([line.split()[3] for line in lines[2:]], dtype=float)
						if train_val == "train":
							lr = np.array([line.split()[4] for line in lines[2:]], dtype=float)

					steps_per_epoch = len(np.where(epochs == 0)[0])
					if steps_per_epoch != 0:
						x = iterations / steps_per_epoch
					else:
						x = iterations

					name = latest_file.replace(data_path, "").replace("history.txt", "")
					if loss_acc == "loss":
						plot_loss_acc(x, train_loss, train_val, loss_acc, plot_path, name, parameters, hold, ax, color, font_size = "big")
					else:
						plot_loss_acc(x, train_acc, train_val, "acc", plot_path, name, parameters, hold, ax, color, font_size = "big")
			
			if train_val == "train" and loss_acc == "loss":
				title = "Train Loss"
				ylabel = "Loss"
			elif train_val == "train" and loss_acc == "acc":
				title = "Train Accuracy"
				ylabel = "Accuracy"
			elif train_val == "val" and loss_acc == "loss":
				title = "Validation Loss"
				ylabel = "Loss"
			elif train_val == "val" and loss_acc == "acc":
				title = "Validation Accuracy"
				ylabel = "Accuracy"
			savepath = plot_path + "input-indipendent_baseline_" + train_val + "_" + loss_acc + ".png"
			plt.xlim([0,50])
			plt.title(title)
			plt.suptitle(None)
			plt.ylabel(ylabel)
			ax.plot([], [], ".", markersize=10, color="blue", label="Real input data")
			ax.plot([], [], ".", markersize=10, color="red", label="Null input data")
			ax.legend()
			plt.draw()
			plt.savefig(savepath, dpi=800, bbox_inches='tight')
			plt.close()
			print("# Figure saved at " + savepath)

	return True

# overfitting single batch
def overfitting_single_batch(data_path, plot_path, date):
	hold = True

	train_val = "train"
	for loss_acc in ["loss", "acc"]:
		fig, ax = plt.subplots()
		color = "blue"
		latest_file = get_latest_file(date, train_val)

		if latest_file:
			with open(latest_file) as file:
				namespace_str = file.readline()
				parameters = parse_namespace_string(namespace_str)

				lines = file.readlines()
				epochs = np.array([line.split()[0] for line in lines[2:]], dtype=int)
				iterations = np.array([line.split()[1] for line in lines[2:]], dtype=int)
				train_loss = np.array([line.split()[2] for line in lines[2:]], dtype=float)
				train_acc = np.array([line.split()[3] for line in lines[2:]], dtype=float)
				if train_val == "train":
					lr = np.array([line.split()[4] for line in lines[2:]], dtype=float)

			steps_per_epoch = len(np.where(epochs == 0)[0])
			if steps_per_epoch != 0:
				x = iterations / steps_per_epoch
			else:
				x = iterations

			name = latest_file.replace(data_path, "").replace("history.txt", "")
			if loss_acc == "loss":
				plot_loss_acc(x, train_loss, train_val, loss_acc, plot_path, name, parameters, hold, ax, color, font_size = "big")
			else:
				plot_loss_acc(x, train_acc, train_val, "acc", plot_path, name, parameters, hold, ax, color, font_size = "big")
		
		if train_val == "train" and loss_acc == "loss":
			title = "Train Loss"
			ylabel = "Loss"
		elif train_val == "train" and loss_acc == "acc":
			title = "Train Accuracy"
			ylabel = "Accuracy"
		savepath = plot_path + "overfitting_single_batch_" + train_val + "_" + loss_acc + ".png"
		plt.xlim([0,2000])
		plt.title(title)
		plt.suptitle(None)
		plt.ylabel(ylabel)
		plt.draw()
		plt.savefig(savepath, dpi=800, bbox_inches='tight')
		plt.close()
		print("# Figure saved at " + savepath)

	return True

# at last we will use 0.1 drop path and 0 smoothing
def droppath_and_smoothing(data_path, plot_path, date_0dp_0sm,	date_01dp_0sm,date_0dp_01sm, date_01dp_01sm):
	hold = True

	train_val = "train"
	for loss_acc in ["loss", "acc"]:
		image_font("small")
		fig, ax = plt.subplots()
		for i, date in enumerate([date_0dp_0sm, date_01dp_0sm, date_0dp_01sm, date_01dp_01sm]):
			if i == 0:
				color = "blue"
			elif i == 1:
				color = "green"
			elif i == 2:
				color = "red"
			elif i == 3:
				color = "orange"
			
			latest_file = get_latest_file(date, train_val)

			if latest_file:
				with open(latest_file) as file:
					namespace_str = file.readline()
					parameters = parse_namespace_string(namespace_str)

					lines = file.readlines()
					epochs = np.array([line.split()[0] for line in lines[2:]], dtype=int)
					iterations = np.array([line.split()[1] for line in lines[2:]], dtype=int)
					train_loss = np.array([line.split()[2] for line in lines[2:]], dtype=float)
					train_acc = np.array([line.split()[3] for line in lines[2:]], dtype=float)
					if train_val == "train":
						lr = np.array([line.split()[4] for line in lines[2:]], dtype=float)

				steps_per_epoch = len(np.where(epochs == 0)[0])
				if steps_per_epoch != 0:
					x = iterations / steps_per_epoch
				else:
					x = iterations

				name = latest_file.replace(data_path, "").replace("history.txt", "")
				if loss_acc == "loss":
					plot_loss_acc(x, train_loss, train_val, loss_acc, plot_path, name, parameters, hold, ax, color, font_size = "small")
				else:
					plot_loss_acc(x, train_acc, train_val, "acc", plot_path, name, parameters, hold, ax, color, font_size = "small")
			
		if train_val == "train" and loss_acc == "loss":
			title = "Train Loss"
			ylabel = "Loss"
		elif train_val == "train" and loss_acc == "acc":
			title = "Train Accuracy"
			ylabel = "Accuracy"
		savepath = plot_path + "droppathVSsmoothing_" + train_val + "_" + loss_acc + ".png"
		plt.xlim([0,2000])
		plt.title(title)
		plt.suptitle(None)
		plt.ylabel(ylabel)
		ax.plot([], [], ".", markersize=10, color="blue", label="No Regularization")
		ax.plot([], [], ".", markersize=10, color="green", label="0.1 Drop Path")
		ax.plot([], [], ".", markersize=10, color="red", label="0.1 Smoothing")
		ax.plot([], [], ".", markersize=10, color="orange", label="Both Regularizations")
		ax.legend(loc='upper right')
		plt.draw()
		plt.savefig(savepath, dpi=800, bbox_inches='tight')
		plt.close()
		print("# Figure saved at " + savepath)

	return True

# overfitting small batch
def overfitting_small_dataset(data_path, plot_path, date):
	hold = True
	print_errorbar = False

	for loss_acc in ["loss", "acc"]:
		font_path = "/nfsroot/home1/homedirs/gusso/miniconda3/envs/convnext/fonts/times.ttf"  # Modify with the actual path
		font_prop = fm.FontProperties(fname=font_path)
		plt.rcParams['font.family'] = font_prop.get_name()
		plt.rcParams.update({
			'font.weight': 'bold',
			'axes.titleweight': 'bold',
			'axes.labelweight': 'bold',
			'font.size': 18,           # Base font size
			'axes.titlesize': 20,      # Title
			'axes.labelsize': 18,      # Axis labels
			'xtick.labelsize': 16,     # X tick labels
			'ytick.labelsize': 16,     # Y tick labels
			'legend.fontsize': 14,     # Legend
			'figure.titlesize': 22     # Suptitle if used
		})
		fig, ax = plt.subplots()
		for train_val in ["val", "training"]:
			if train_val == "training":
				latest_file = get_latest_file(date, "train")
				color = "red"
			else:
				latest_file = get_latest_file(date, "val")
				color = "blue"

			if latest_file:
				with open(latest_file) as file:
					namespace_str = file.readline()
					parameters = parse_namespace_string(namespace_str)

					lines = file.readlines()
					epochs = np.array([line.split()[0] for line in lines[2:]], dtype=int)
					iterations = np.array([line.split()[1] for line in lines[2:]], dtype=int)
					train_loss = np.array([line.split()[2] for line in lines[2:]], dtype=float)
					train_acc = np.array([line.split()[3] for line in lines[2:]], dtype=float)
					if train_val == "training":
						train_acc = train_acc*100
					if train_val == "train":
						lr = np.array([line.split()[4] for line in lines[2:]], dtype=float)

				steps_per_epoch = len(np.where(epochs == 0)[0])
				if steps_per_epoch != 0:
					x = iterations / steps_per_epoch
				else:
					x = iterations

				name = latest_file.replace(data_path, "").replace("history.txt", "")
				if loss_acc == "loss":
					plot_loss_acc(x, train_loss, train_val, loss_acc, plot_path, name, parameters, hold, ax, color, print_errorbar, font_size = "big")
				else:
					plot_loss_acc(x, train_acc, train_val, "acc", plot_path, name, parameters, hold, ax, color, print_errorbar, font_size = "big")
			
		if loss_acc == "loss":
			title = "Training and Validation Loss"
			ylabel = "Loss"
		elif loss_acc == "acc":
			title = "Training and Validation Accuracy"
			ylabel = "Accuracy"
		savepath = plot_path + "overfitting_small_dataset_" + loss_acc + ".png"
		plt.xlim([0,300])
		plt.title(title)
		plt.suptitle(None)
		plt.ylabel(ylabel)
		ax.plot([], [], ".", markersize=10, color="red", label="Training")
		ax.plot([], [], ".", markersize=10, color="blue", label="Validation")
		ax.legend(loc='lower left')
		plt.draw()
		plt.savefig(savepath, dpi=800, bbox_inches='tight')
		plt.close()
		print("# Figure saved at " + savepath)

	return True

# learning rate scheduler for 50 epochs
def cosine_lr(data_path, plot_path, date):
	hold = True

	train_val = "train"
	fig, ax = plt.subplots()
	color = "blue"
	latest_file = get_latest_file(date, train_val)

	if latest_file:
		with open(latest_file) as file:
			namespace_str = file.readline()
			parameters = parse_namespace_string(namespace_str)

			lines = file.readlines()
			epochs = np.array([line.split()[0] for line in lines[2:]], dtype=int)
			iterations = np.array([line.split()[1] for line in lines[2:]], dtype=int)
			train_loss = np.array([line.split()[2] for line in lines[2:]], dtype=float)
			train_acc = np.array([line.split()[3] for line in lines[2:]], dtype=float)
			if train_val == "train":
				lr = np.array([line.split()[4] for line in lines[2:]], dtype=float)

		steps_per_epoch = len(np.where(epochs == 0)[0])
		if steps_per_epoch != 0:
			x = iterations / steps_per_epoch
		else:
			x = iterations

		name = latest_file.replace(data_path, "").replace("history.txt", "")
		plot_loss_acc(x, lr, train_val, "lr", plot_path, name, parameters, hold, ax, color, font_size = "big")

	title = "Learning Rate Schedule"
	ylabel = "Learning Rate"
	savepath = plot_path + "cosine_lr_" + train_val + "_" + "lr" + ".png"

	formatter = ScalarFormatter(useMathText=True)
	formatter.set_scientific(True)
	formatter.set_powerlimits((-2, 2))
	ax.yaxis.set_major_formatter(formatter)

	plt.xlim([0,50])
	plt.title(title)
	plt.suptitle(None)
	plt.ylabel(ylabel)
	plt.xlabel("Epochs")
	plt.draw()
	plt.savefig(savepath, dpi=800, bbox_inches='tight')
	plt.close()
	print("# Figure saved at " + savepath)

	return True

# ValidR plot comparison
def ValidR_plot_comparison(data_path, plot_path, ValidRh0_input, ValidRP_input, ValidRe_input, ValidRap_input):
	hold = True

	for train_val in ["train", "val"]:
		for loss_acc in ["loss", "acc"]:
			image_font("small")
			fig, ax = plt.subplots()
			for i, date in enumerate([ValidRh0_input, ValidRP_input, ValidRe_input, ValidRap_input]):
				if i == 0:
					color = "blue"
				elif i == 1:
					color = "green"
				elif i == 2:
					color = "red"
				elif i == 3:
					color = "orange"

				latest_file = get_latest_file(date, train_val)

				if latest_file:
					with open(latest_file) as file:
						namespace_str = file.readline()
						parameters = parse_namespace_string(namespace_str)

						lines = file.readlines()
						epochs = np.array([line.split()[0] for line in lines[2:]], dtype=int)
						iterations = np.array([line.split()[1] for line in lines[2:]], dtype=int)
						train_loss = np.array([line.split()[2] for line in lines[2:]], dtype=float)
						train_acc = np.array([line.split()[3] for line in lines[2:]], dtype=float)
						if train_val == "train":
							lr = np.array([line.split()[4] for line in lines[2:]], dtype=float)

					steps_per_epoch = len(np.where(epochs == 0)[0])
					if steps_per_epoch != 0:
						x = iterations / steps_per_epoch
					else:
						x = iterations

					name = latest_file.replace(data_path, "").replace("history.txt", "")
					if loss_acc == "loss":
						plot_loss_acc(x, train_loss, train_val, loss_acc, plot_path, name, parameters, hold, ax, color, font_size = "small")
					else:
						plot_loss_acc(x, train_acc, train_val, "acc", plot_path, name, parameters, hold, ax, color, font_size = "small")
			
			if train_val == "train" and loss_acc == "loss":
				title = "Train Loss"
				ylabel = "Loss"
			elif train_val == "train" and loss_acc == "acc":
				title = "Train Accuracy"
				ylabel = "Accuracy"
			elif train_val == "val" and loss_acc == "loss":
				title = "Validation Loss"
				ylabel = "Loss"
			elif train_val == "val" and loss_acc == "acc":
				title = "Validation Accuracy"
				ylabel = "Accuracy"
			savepath = plot_path + "ValidR_comparison_" + train_val + "_" + loss_acc + ".png"
			plt.xlim([0,50])
			plt.title(title)
			plt.suptitle(None)
			plt.ylabel(ylabel)
			ax.plot([], [], ".", markersize=10, color="blue", label=r"ValidR-$h_0$")
			ax.plot([], [], ".", markersize=10, color="green", label=r"ValidR-$P$")
			ax.plot([], [], ".", markersize=10, color="red", label=r"ValidR-$e$")
			ax.plot([], [], ".", markersize=10, color="orange", label=r"ValidR-$a_p$")
			ax.legend()
			plt.draw()
			plt.savefig(savepath, dpi=800, bbox_inches='tight')
			plt.close()
			print("# Figure saved at " + savepath)

	return True

# ValidR performance comparison
def ValidR_performances(data_path, ValidRh0_input, ValidRP_input, ValidRe_input, ValidRap_input):
	suffix = "_val_history.txt"

	print("\n\n\n")
	print("||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")
	print("|||||||||||||||||||        PRINT PERFORMANCES        |||||||||||||||||||")
	print("||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")
	process_val_history(os.path.join(data_path, ValidRh0_input + suffix))
	process_val_history(os.path.join(data_path, ValidRP_input + suffix))
	process_val_history(os.path.join(data_path, ValidRe_input + suffix))
	process_val_history(os.path.join(data_path, ValidRap_input + suffix))

# ValidG plot comparison
def ValidG_plot_comparison(data_path, plot_path, ValidGh0_input, ValidGP_input, ValidGe_input, ValidGap_input):
	hold = True

	for train_val in ["train", "val"]:
		for loss_acc in ["loss", "acc"]:
			image_font("small")
			fig, ax = plt.subplots()
			for i, date in enumerate([ValidGh0_input, ValidGP_input, ValidGe_input, ValidGap_input]):
				if i == 0:
					color = "blue"
				elif i == 1:
					color = "green"
				elif i == 2:
					color = "red"
				elif i == 3:
					color = "orange"

				latest_file = get_latest_file(date, train_val)

				if latest_file:
					with open(latest_file) as file:
						namespace_str = file.readline()
						parameters = parse_namespace_string(namespace_str)

						lines = file.readlines()
						epochs = np.array([line.split()[0] for line in lines[2:]], dtype=int)
						iterations = np.array([line.split()[1] for line in lines[2:]], dtype=int)
						train_loss = np.array([line.split()[2] for line in lines[2:]], dtype=float)
						train_acc = np.array([line.split()[3] for line in lines[2:]], dtype=float)
						if train_val == "train":
							lr = np.array([line.split()[4] for line in lines[2:]], dtype=float)

					steps_per_epoch = len(np.where(epochs == 0)[0])
					if steps_per_epoch != 0:
						x = iterations / steps_per_epoch
					else:
						x = iterations

					name = latest_file.replace(data_path, "").replace("history.txt", "")
					if loss_acc == "loss":
						plot_loss_acc(x, train_loss, train_val, loss_acc, plot_path, name, parameters, hold, ax, color, font_size = "small")
					else:
						plot_loss_acc(x, train_acc, train_val, "acc", plot_path, name, parameters, hold, ax, color, font_size = "small")
			
			if train_val == "train" and loss_acc == "loss":
				title = "Train Loss"
				ylabel = "Loss"
			elif train_val == "train" and loss_acc == "acc":
				title = "Train Accuracy"
				ylabel = "Accuracy"
			elif train_val == "val" and loss_acc == "loss":
				title = "Validation Loss"
				ylabel = "Loss"
			elif train_val == "val" and loss_acc == "acc":
				title = "Validation Accuracy"
				ylabel = "Accuracy"
			savepath = plot_path + "ValidG_comparison_" + train_val + "_" + loss_acc + ".png"
			plt.xlim([0,50])
			plt.title(title)
			plt.suptitle(None)
			plt.ylabel(ylabel)
			ax.plot([], [], ".", markersize=10, color="blue", label=r"ValidG-$h_0$")
			ax.plot([], [], ".", markersize=10, color="green", label=r"ValidG-$P$")
			ax.plot([], [], ".", markersize=10, color="red", label=r"ValidG-$e$")
			ax.plot([], [], ".", markersize=10, color="orange", label=r"ValidG-$a_p$")
			ax.legend()
			plt.draw()
			plt.savefig(savepath, dpi=800, bbox_inches='tight')
			plt.close()
			print("# Figure saved at " + savepath)

	return True

# ValidG performance comparison
def ValidG_performances(data_path, ValidGh0_input, ValidGP_input, ValidGe_input, ValidGap_input):
	suffix = "_val_history.txt"

	print("\n\n\n")
	print("||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")
	print("|||||||||||||||||||        PRINT PERFORMANCES        |||||||||||||||||||")
	print("||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")
	process_val_history(os.path.join(data_path, ValidGh0_input + suffix))
	process_val_history(os.path.join(data_path, ValidGP_input + suffix))
	process_val_history(os.path.join(data_path, ValidGe_input + suffix))
	process_val_history(os.path.join(data_path, ValidGap_input + suffix))


# Change "DATE-HOUR" with the date and hour of the relative history file (e.g. "20250412-125050") saved in the history folder

loss_at_init()
plot_zero_input(data_path, plot_path, date_zero_input = "DATE-HOUR", date_real_input = "DATE-HOUR")
overfitting_single_batch(data_path, plot_path, date = "DATE-HOUR")
droppath_and_smoothing(data_path, plot_path, date_0dp_0sm = "DATE-HOUR", date_01dp_0sm = "DATE-HOUR", date_0dp_01sm = "DATE-HOUR", date_01dp_01sm = "DATE-HOUR")
overfitting_small_dataset(data_path, plot_path, date = "DATE-HOUR")
cosine_lr(data_path, plot_path, date = "DATE-HOUR")
ValidR_plot_comparison(data_path, plot_path, ValidRh0_input = "DATE-HOUR", ValidRP_input = "DATE-HOUR", ValidRe_input = "DATE-HOUR", ValidRap_input = "DATE-HOUR")
ValidR_performances(data_path, ValidRh0_input = "DATE-HOUR", ValidRP_input = "DATE-HOUR", ValidRe_input = "DATE-HOUR", ValidRap_input = "DATE-HOUR")
ValidG_plot_comparison(data_path, plot_path, ValidGh0_input = "DATE-HOUR", ValidGP_input = "DATE-HOUR", ValidGe_input = "DATE-HOUR", ValidGap_input = "DATE-HOUR")
ValidG_performances(data_path, ValidGh0_input = "DATE-HOUR", ValidGP_input = "DATE-HOUR", ValidGe_input = "DATE-HOUR", ValidGap_input = "DATE-HOUR")
