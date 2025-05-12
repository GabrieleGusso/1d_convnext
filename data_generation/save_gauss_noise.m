clear all;
addpath(genpath('/PATH/1d_convnext/data_generation'))
delete(gcp('nocreate'))

% Script to generate the different train and validation datasets in white Gaussian noise
% Substitute PATH with your local path
% Chose a name_dir of your preference, while remembering to change it while varying the value (var_value) of the current parameter (var_param)

name_dir = 'THESIS_gaussnoise_f_70-270_h0_5-50e25_P_19_ecc_0_asini_1_cosi_0';
var_value = [5e-25 50e-25]; % min and max values of the varying parameter
var_param = "h0";
init_gusso(name_dir)

train_val_ratio = 0.95; % train samples over val samples ratio
n_pulsars = 2000; % (standard: 2000) - per ogni 1000 pulsars (1000 signal + 1000 noise) ci mette 1:40h ed occupa 2gb
freq_start = 70; % in Hz (standard: 70)
bandwidth = 0.1; % in Hz (standard: 0.1)
time_band = 15; % days to take for each time series
time_tot_bands = 0; % number of time bands to take (if 0 takes the whole run)

band_samples = n_pulsars * 2;
start_O3a = 58574;
end_O3a = 58756;
start_O3b = 58789;
end_O3b = 58935;
time_O3a = fix((end_O3a - start_O3a) / time_band);
time_O3b = fix((end_O3b - start_O3b) / time_band);
if time_tot_bands == 0
    time_tot_bands = fix((end_O3a - start_O3a) / time_band) + fix((end_O3b - start_O3b) / time_band);
elseif time_tot_bands <= time_O3b+time_O3a
    time_tot_bands;
else
    error("Time series is out of O3a and O3b boundaries. For a time_band of " + num2str(time_band) + " days the maximum value of time_tot_bands is: " + num2str(time_O3a+time_O3b))
end
tot_samples = band_samples * time_tot_bands;
val_samples = round((1-train_val_ratio)*band_samples);
freq_stop = freq_start + n_pulsars * bandwidth;

fprintf("\n# Generating %i samples from %i Hz to %i Hz divided in bands of %.2f Hz:\n", [tot_samples, freq_start, freq_stop, bandwidth])
fprintf("# - %i train signal samples\n", tot_samples - val_samples * time_tot_bands)
fprintf("# - %i train noise samples\n", tot_samples - val_samples * time_tot_bands)
fprintf("# - %i val signal samples\n", val_samples * time_tot_bands)
fprintf("# - %i val noise samples\n\n", val_samples * time_tot_bands)

%% PARAMETERS
ant             = ligol;
DEVBSD 			= '/storage';
data_tobeused   = 'C01_GATED_SUB60HZ_O3';
det             = ant.name; % 'ligol'
hl              = load('data/tables/HL_ASDs.mat'); % import the amplitude spectral density of H and L

max_workers = time_tot_bands;
parpool(max_workers);
parfor time_idx = 0:time_tot_bands-1

    % must go from 01-Apr-2019 (58574) to 30-Sep-2019 (58756) and from 01-Nov-2019 (58788) to 27-Mar-2020 (58935)
    if time_idx < time_O3a
        start_mjd = start_O3a + time_idx * time_band;
    elseif time_idx >= time_O3a && time_idx-time_O3a < time_O3b
        start_mjd = start_O3b + (time_idx-time_O3a) * time_band;
    else
        error("Time series is out of O3a and O3b boundaries. For a time_band of " + num2str(time_band) + " days the maximum value of time_tot_bands is: " + num2str(time_O3a+time_O3b))
    end
    end_mjd   = start_mjd + time_band;
    fprintf("# start_mjd and end_mjd for the band n. %i: %i %i\n", [time_idx+1, start_mjd, end_mjd])

    %% GENERATE PULSAR PARAMETERS
    % deve essere salvato diversamente per ogni nuovo PERIODO di dati
    gen_data_parameters(n_pulsars, freq_start, bandwidth, start_mjd, end_mjd);
	rng('shuffle');
    min_idx = 1;
	max_idx = n_pulsars;
	val_idx = randi([min_idx max_idx],1,val_samples);

    %% IDX
    database_idx_offset = time_idx * band_samples;
    database_idx = 1;
    database_idx = int32(database_idx / 2) * 2 - 1;
    start = double(int32(database_idx/2 + 0.1));
    database_idx = database_idx + database_idx_offset;

    %% TEST NOISE
    main_noise   = bsd_lego_gusso(DEVBSD,det,data_tobeused,[start_mjd end_mjd],[freq_start freq_start+bandwidth],1); 
    main_noise   = cut_bsd(main_noise, [start_mjd end_mjd]);

    for kpuls = start:n_pulsars

        %% BSD
        fstart  = freq_start + (kpuls-1) * bandwidth;
        fend    = fstart + bandwidth;

        %% NOISE
        noise = gd_gen_noise(main_noise, fstart, bandwidth, hl, time_band)

        %% TRAIN or VAL
        if ismember(kpuls, val_idx)
			train_val = 'val';
		else
			train_val = 'train';
        end

        %% SIGNAL
        save_signal(kpuls, database_idx, train_val, start_mjd, end_mjd, noise, name_dir); % saves kpuls' spectrum data
        if time_idx == 0
            fprintf("# [%d/%d] signal pulsar %d\n", [database_idx*time_tot_bands, band_samples*time_tot_bands, kpuls])
        end
        database_idx = database_idx + 1;

        %% NOISE
        save_noise(kpuls, database_idx, train_val, noise, name_dir);
        if time_idx == 0
            fprintf("# [%d/%d] noise pulsar %d\n", [database_idx*time_tot_bands, band_samples*time_tot_bands, kpuls])
        end
        database_idx = database_idx + 1;
    end
end

fprintf("# Dataset generation completed\n")
delete(gcp('nocreate'))
