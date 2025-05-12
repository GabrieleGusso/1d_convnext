function init_gusso(name_dir)

    %     name_dir   name of the directory that will be created to save the new data

    % by G.Gusso - gabriele.gusso@roma1.infn.it
    % Department of Physics - Sapienza University - Rome

    % Initial settings of the driver, such as paths and folders
    % TO MODIFY WITH YOUR PATH
    
    addpath(genpath('/PATH/Snag'))
    addpath(genpath('/PATH/BSD'))

    if exist(['/PATH/1d_convnext/database/' name_dir], 'dir')
        rmdir(['/PATH/1d_convnext/database/' name_dir], 's')
    end
    NEW_FOLDERS = ["", "/train", "/train/signal", "/train/noise", "/val", "/val/signal", "/val/noise", "/plots", "/plots/train", "/plots/train/signal", "/plots/train/noise", "/plots/val", "/plots/val/signal", "/plots/val/noise"];
    new_path = "/PATH/1d_convnext/database/" + string(name_dir);
    for i = 1:length(NEW_FOLDERS)
        folder = NEW_FOLDERS(i);
        folder_path = convertStringsToChars(new_path + folder);
        if ~exist(folder_path, 'dir')
            mkdir(folder_path);
        end
    end

    path = "/PATH/1d_convnext/data_generation";
    FOLDERS     = ["/plots/spectrum", "/data/spectrum", "/plots/peakmap"    , "/data/peakmap"          ];
    SUB_FOLDERS = ["/real_noise"    , "/gauss_noise"  , "/gauss_noise_paola", "/gauss_noise_comparison"];
    TRIGGER     = [[1,1,1,1]        ; [1,1,0,0]       ; [1,1,0,0]           ; [1,0,0,0]                ];
    % TRIGGER refers to which FOLDERS to add or not the corresponding SUB_FOLDERS element,
    % for example for gauss_noise I have [1,1,0,0], so I create the sub-folders only in the first two FOLDERS
    
    for i = 1:length(FOLDERS)
        folder = FOLDERS(i);
        folder_path = convertStringsToChars(path + folder);
        if ~exist(folder_path, 'dir')
            mkdir(folder_path);
        end
        
        for j = 1:length(SUB_FOLDERS)
            if TRIGGER(j,i) == 1
                sub_folder = SUB_FOLDERS(j);
                sub_folder_path = convertStringsToChars(path + folder + sub_folder);
                if ~exist(sub_folder_path, 'dir')
                    mkdir(sub_folder_path);
                end
            end
        end
    end
