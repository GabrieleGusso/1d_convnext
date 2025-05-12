function save_signal(kpuls, database_idx, train_val, start_mjd, end_mjd, noise, name_dir)

	%    kpuls   		number of the current pulsars
	%    database_idx   progressive database index to save data
	%    train_val   	train or validation dataset
	%    start_mjd   	initial modified julian date for BSD
	%    end_mjd     	ending modified julian date for BSD
	%    noise  		noise gd
	%    name_dir     	name of the directory where to save the data
	
	% by G. Gusso - gabriele.gusso@roma1.infn.it
	% Department of Physics - Sapienza University - Rome

	%% PULSAR
	sour    	= read_cwinj_table_gusso("/nfsroot/home1/homedirs/gusso/Tesi_codici/MATLAB/data/tables/PulsarParametersMjd" + num2str(start_mjd) + "to" + num2str(end_mjd) + ".dat", kpuls); % per le software injections
	radsec  	= 1;
	einsteinFac = 9.999999682655225e-21;
	H0          = sour.h; % to go from H0 to LIGO standard h0 see read_cwinj_table(), eq.5 di articolo mio e di AS
	A           = 0.5*H0/einsteinFac; % amplitude of the injected signal! Devi moltiplicare per 1/2 perche' i BSD sono complessi ed usano lo spettro monolatero; 

	%% INJECTION
	if isfield(sour,'ecc') && sour.ecc <= 0.1
		EccAnomOption = 0; % low-ecc case
	elseif isfield(sour,'ecc') && sour.ecc > 0.1
		EccAnomOption = 1; % high-ecc case
	elseif ~isfield(sour,'ecc')
		EccAnomOption = 'none';
	end
	[out Hp Hc sig ph0] = bsd_softinj_re_mod_isobin(noise,sour,A,0,radsec,EccAnomOption);

	%% NON-CORRECTED SPECTRUM
	path = ['/data01/gusso/convnext/database/' name_dir];
	bsd_pows_gusso_ssd(path, train_val, 'signal', out, kpuls, database_idx);

	clear sour out Hp Hc sig ph0
end
