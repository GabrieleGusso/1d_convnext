function out = gd_gen_noise(main_noise, fstart, bandwidth, hl, T_obs)

	%    main_noise   reference gd for noise
	%    fstart       initial frequency of the first frequency band
	%    bandwidth    frequency band width
	%    hl           detector information
	%    T_obs        time of observation in days

	% by G. Gusso - gabriele.gusso@roma1.infn.it
	% Department of Physics - Sapienza University - Rome

	N				= n_gd(main_noise); % 129600 for 15 days
	ASD_freq        = hl.Info.frL;
	ASD             = hl.Info.ASDL;
	[~,idx]         = min(abs(ASD_freq - (fstart + bandwidth / 2)));
	einsteinFac 	= 9.999999682655225e-21;
	
	ASD_band_mean   = ASD(min(idx)) / einsteinFac;
	gaussian_noise  = (ASD_band_mean * randn(N, 1)) / sqrt(T_obs); % gaussian noise h(t) ~ √(Sh(f)/T_obs) with Sh(f) = ASD_band
	
	control_vars 		= cont_gd(main_noise);
	control_vars.inifr  = fstart;
	out = edit_gd(main_noise, 'y', gaussian_noise, 'cont', control_vars);

end
