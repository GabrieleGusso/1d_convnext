# by G. Gusso - gabriele.gusso@roma1.infn.it
# Department of Physics - Sapienza University - Rome

# Change PATH with your path (also for the font_path)

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import os, glob, re

date = None
# date = "20250411-131816" # if you want plots for a specific execution
hold = False
train_val = "all"
data_path = "/PATH/1d_convnext/results/history/"
plot_path = "/PATH/1d_convnext/neural_network/plots/"


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


def plot_loss_acc(x, y, train_val, loss_acc, plot_path, name, parameters, hold = False, ax = None, color = "blue", print_errorbar = True):
    savepath = plot_path + name + loss_acc + ".png"
    if loss_acc == "loss":
        norm = np.log(2)
        y = y / norm

    if ax == None:
        fig, ax = plt.subplots()
    font_path = "/PATH/miniconda3/envs/convnext/fonts/times.ttf"  # Modify with the actual path
    font_prop = fm.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = font_prop.get_name()
    plt.rcParams.update({
        'font.size': 14,           # Base font size
        'axes.titlesize': 16,      # Title
        'axes.labelsize': 14,      # Axis labels
        'xtick.labelsize': 12,     # X tick labels
        'ytick.labelsize': 12,     # Y tick labels
        'legend.fontsize': 12,     # Legend
        'figure.titlesize': 18     # Suptitle if used
    })

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


def plot_main(
    date: str, train_val: str, hold: bool, data_path: str, plot_path: str, subtitle: str = None
):
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
            if "train" in train_val:
                lr = np.array([line.split()[4] for line in lines[2:]], dtype=float)

        steps_per_epoch = len(np.where(epochs == 0)[0])
        if steps_per_epoch != 0:
            x = iterations / steps_per_epoch
        else:
            x = iterations

        name = latest_file.replace(data_path, "").replace("history.txt", "")
        plot_loss_acc(x, train_loss, train_val, "loss", plot_path, name, parameters, hold)
        plot_loss_acc(x, train_acc, train_val, "acc", plot_path, name, parameters, hold)
        if "train" in train_val:
            plot_loss_acc(x, lr, train_val, "lr", plot_path, name, parameters, hold)

        return parameters


def plot_graphs(date, train_val, data_path, plot_path):
    if train_val == "all":
        parameters = plot_main(date, "train", hold, data_path, plot_path)
        plot_main(date, "val", hold, data_path, plot_path)
        plot_main(date, "iter", hold, data_path, plot_path)
    else:
        parameters = plot_main(date, train_val, hold, data_path, plot_path)

    print("")
    for param, value in parameters.__dict__.items():
        print(f"parameters.{param} = {value}")
    print("\n# Figures saved at " + str(plot_path) + "\n")

    return True


plot_graphs(date, train_val, data_path, plot_path)
