function save_noise(kpuls, database_idx, train_val, noise, name_dir)

	%    kpuls   		number of the current pulsars
	%    database_idx   progressive database index to save data
	%    train_val   	train or validation dataset
	%    noise  		noise gd
	%    name_dir     	name of the directory where to save the data
	
	% by G. Gusso - gabriele.gusso@roma1.infn.it
	% Department of Physics - Sapienza University - Rome

	%% NON-CORRECTED SPECTRUM
	path = ['/data01/gusso/convnext/database/' name_dir];
	bsd_pows_gusso_ssd(path, train_val, 'noise', noise, kpuls, database_idx);
end
