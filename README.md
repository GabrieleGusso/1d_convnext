# README

The current repository is used in the Master thesis work "A novel machine-learning based method to search for continuous-wave signals from neutron stars in binary systems" of Gabriele Gusso, Paola Leaci and Federico Muciaccia.

We present a novel machine learning procedure for the detection of continuous gravitational waves (CWs) emitted by neutron stars in binary systems. The method targets the characteristic double-horn spectral signature arising from orbital Doppler modulation, using a deep convolutional neural network adapted from the ConvNeXt architecture. The network classifies Earth-Doppler-corrected power spectra into signal-plus-noise or noise-only categories, operating within a semi-coherent detection scheme. Our analysis focuses on a fixed sky position corresponding to Scorpius X-1 and assumes negligible spin-down.

The model is trained using both white Gaussian noise and real O3 data from the LIGO-Virgo-KAGRA Collaboration. We inject 42000 software-simulated CW signals for training, spanning a restricted parameter space due to memory constraints:

f ∈ [70, 270] Hz, h0 ∈ [5, 50]×1e-25, P ∈ [10, 48] h, e ∈ [0, 0.49], ap ∈ [1, 1.9] ls.

The use of relatively strong signals (h0) ensures reliable detection during training, facilitating the validation of the method.

We assess model performance on 80 independent test sets, each with synthetic signals injected into either white Gaussian or real O3 noise. These test datasets cover:

f ∈ [70, 80] Hz, h0 ∈ [5, 50]×1e-25, P ∈ [5, 50] h, e ∈ [0, 0.45], ap ∈ [0.5, 5] ls.

The network shows strong generalization. Models trained on Gaussian noise achieve over 95 % accuracy across both Gaussian and O3 noise. When trained directly on O3 data, accuracy exceeds 98 % on matched test sets.

## REPOSITORY ORGANIZATION

The present repository is organized into three main directories, each serving a distinct purpose:

1. data_generation – Contains the scripts and tools required for dataset creation;
2. database – Stores the generated datasets and associated outputs;
3. neural_network – Includes the implementation and evaluation of the neural network models.

The directory structure is defined as follows (directories marked with an asterisk are generated automatically during execution):

database
 └── dataset_name*        # Automatically generated dataset directory
     ├── plots*           # Automatically generated visualizations
     ├── train*           # Training data
     └── val*             # Validation data
 └── results              # Output results
     └── history          # Training history and logs
data_generation
 ├── functions            # Supporting functions for data generation
 ├── my_Snag              # Custom modules or scripts
 └── data
     └── tables           # Raw input tables or metadata
neural_network
 ├── models               # Model architectures and configurations
 ├── plots                # Evaluation and diagnostic plots
 └── TEST_eval            # Evaluation results
     ├── TrainG_ap
     ├── TrainG_e
     ├── TrainG_h0
     ├── TrainG_P
     ├── TrainR_ap
     ├── TrainR_e
     ├── TrainR_h0
     └── TrainR_P

## PREREQUISITES

### MATLAB

MATLAB version R2023b
Download Snag (vers. 2, rel. 25 November 2022) from https://www.sergiofrasca.net/wp-content/wwwContent/_Phys/snag.htm

### Python

Install MiniConda, Python and various dependencies:

**conda 24.3.0**  
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh  
bash Miniconda3-latest-Linux-x86_64.sh

**python 3.12.2**  
conda install python

**numpy 1.26.4**  
conda install numpy

**matplotlib 3.8.0**  
conda install matplotlib

**pytorch 2.2.1**  
conda install pytorch torchvision torchaudio cpuonly -c pytorch

**torchmetrics 1.4.0.post0**  
conda install -c conda-forge torchmetrics

**torchvision 0.18.1**
conda install pytorch torchvision -c pytorch -c conda-forge

**pytorch-cuda 12.1**  
sudo apt-get -y torchvision torchaudio pytorch-cuda -c pytorch -c nvidia

**cudnn 8.9.2.26**  
conda install -c anaconda cudnn

**scikit-learn 1.4.1.post1**  
conda install -c conda-forge scikit-learn

**pandas**  
conda install pandas

**num_tools 0.11.0**
conda install -c conda-forge enum_tools

**receptive_field**
pip install git+https://github.com/Fangyh09/pytorch-receptive-field.git

**timm**
conda install conda-forge::timm

**to install the other dependencies**
python3 -m pip install --user numpy matplotlib torch scikit-learn

## EXECUTION

To reproduce the experimental results and execute the provided codebase, we recommend following the steps below in sequential order:

1. clone or download the repository;
2. recreate the directory structure as illustrated in the repository documentation to ensure consistent repository organization;
3. install all required dependencies and packages as listed in the prerequisites;
4. in all relevant scripts, replace the placeholder string "PATH" with the absolute path to your local working directory;
5. generate the training and validation datasets using the provided MATLAB scripts: run save_gauss_noise.m and save_real_noise.m (e.g., for dataset h0) to simulate both Gaussian and real-world noise conditions;
6. train and validate the model using the shell script convnext1d_TRAIN.sh (e.g., for dataset h0) on the previously generated datasets;
7. reproduce the training and validation results using chapter_training.py, ensuring that all timestamp placeholders (i.e., "DATE-HOUR") in function calls are replaced with the corresponding identifiers given after training;
8. generate the test datasets using: save_gauss_noise_tests.m and save_real_noise_tests.m (e.g., for dataset h0), following the same conventions used for training data;
9. evaluate the trained model on the test datasets by executing convnext1d_TEST.sh (e.g., for dataset h0);
10. reproduce the test results using chapter_testing.py, as with training, ensuring that all timestamp references are correctly substituted to match your test run identifiers.
