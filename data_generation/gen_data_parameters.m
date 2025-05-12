function gen_data_parameters(n_pulsars, freq_start, bandwidth, start_mjd, end_mjd, var_value, var_param)

	%    n_pulsars   total number of pulsars to generate parameters for
	%    freq_start  initial frequency of the first frequency band
	%    bandwidth   frequency band width
	%    start_mjd   initial modified julian date for BSD
	%    end_mjd     ending modified julian date for BSD
	%    var_value   [min, max] value of the parameter that is being varied during the generation of the dataset
	%    var_param   name of the parameter that is being varied during the generation of the dataset
	%                [strain amplitude "h0", orbital period "P", orbital eccentricity "e", projected semi-major axis "ap"]
	
	% by G. Gusso - gabriele.gusso@roma1.infn.it
	% Department of Physics - Sapienza University - Rome

	addpath(genpath("/PATH/1d_convnext/data_generation"));
 	% Substitute PATH with your local path

	file_name = "/PATH/1d_convnext/data_generation/data/tables/PulsarParametersMjd" + num2str(start_mjd) + "to" + num2str(end_mjd) + ".dat";
	if exist(file_name, "file")
		delete(file_name)
	end
	file = fopen(file_name,"w");
	fprintf(file, "varName,varData\n");

	h_min = 2e-24;
	h_max = 2e-24;
	P_min = 68023.92;
	P_max = 68023.92;
	ecc_min = 0;
	ecc_max = 0;
	asini_min = 1;
	asini_max = 1; 
	cosi_min = 0;
	cosi_max = 0;

	if var_param == "h0"
		h_min = var_value(1);
		h_max = var_value(2);
	elseif var_param == "P"
		P_min = var_value(1);
		P_max = var_value(2);
	elseif var_param == "e"
		ecc_min = var_value(1);
		ecc_max = var_value(2);
	elseif var_param == "ap"
		asini_min = var_value(1);
		asini_max = var_value(2);
	end

	freq_extra_belt = 0.001;

	for i = 1:n_pulsars
		h0 = h_min+rand(1)*(h_max-h_min); 
		cosi = cosi_min+rand(1)*(cosi_max-cosi_min); 
		asini = asini_min+rand(1)*(asini_max-asini_min); 
		period = P_min+rand(1)*(P_max-P_min); 
		ecc = ecc_min+rand(1)*(ecc_max-ecc_min); 

		freq_max = freq_start + bandwidth*i;
		freq_belt = freq_max * ( ((2 * pi) / period * asini / (1 - ecc))) + freq_extra_belt;
		if freq_belt > bandwidth/2
			error('The frequency belt oversizes the half of the frequency band width, choose different parameters.')
		end
		freq_ini = freq_start + bandwidth*(i-1) + freq_belt;
		freq_end = freq_ini + bandwidth - 2*freq_belt;
		freq = freq_ini+rand(1)*(freq_end-freq_ini);

		fprintf(file, "\n");
		printVar(file,"Pulsar " + num2str(i),[]);
		printVar(file,"Alpha",4.27569814767);
		printVar(file,"Delta",-0.25062540207);
		printVar(file,"h0",h0); 
		printVar(file,"cosi",cosi); 
		printVar(file,"psi",0);
		printVar(file,"phi0",0);
		printVar(file,"Freq",freq);
		printVar(file,"orbitasini",asini); 
		printVar(file,"orbitEcc",ecc);
		printVar(file,"orbitTp",731163327);
		printVar(file,"orbitPeriod",period); 
		printVar(file,"orbitArgp",0);
		printVar(file,"f1dot",0);
		printVar(file,"refTime",731163327);
	end

	fclose(file);
end
