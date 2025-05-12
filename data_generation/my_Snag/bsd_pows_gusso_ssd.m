function sn=bsd_pows_gusso_ssd(path, train_val, signal_noise, in, kpuls, database_idx)
    % power spectrum creation and saving
    %
    %      pows=bsd_pows_gusso(in,save_info)
    %
    %    in         input bsd
    %    save_info  info to save the power spectrum [kpuls, correction]
    %               correction: 0 no corr, 1 spin-down and Doppler corr, 2 spin-down, Doppler and binary corr
    
    % by G.Gusso - gabriele.gusso@roma1.infn.it
    % Department of Physics - Sapienza University - Rome

    pows = bsd_pows(in,1);
    sn = single(y_gd(pows))';
    % size(sn)

    % Plot save
    fsz = 24;
    figure; semilogy(pows);
    set(gca, 'TickLabelInterpreter', 'latex', 'FontSize', fsz-4)
    xlabel('Frequency [Hz]', 'Interpreter', 'latex', 'FontSize', fsz)
    ylabel('Amplitude Spectral Density $[1/\sqrt{\mathrm{Hz}}]$', 'Interpreter', 'latex', 'FontSize', fsz)
    % xlim([76.3 76.55]); ylim([1e-8 1e-4]);


    title('Rescaled ASD', 'Interpreter', 'latex', 'FontSize', fsz);
    savepath=[path '/plots/' train_val '/' signal_noise '/sample_' num2str(database_idx) '_fake_pulsar_' num2str(kpuls) '_spectrum_noncorr.png']; saveas(gcf, savepath)
    % fprintf("\n# Non corrected power spectrum figure saved at " + savepath + "\n")

    % Data save
    savepath=[path '/' train_val '/' signal_noise '/sample_' num2str(database_idx) '_fake_pulsar_' num2str(kpuls) '_pows_noncorr.mat']; save(savepath, "sn")
    % fprintf("# Non corrected power spectrum data saved at " + savepath + "\n")

    close all
    clear pows sn
