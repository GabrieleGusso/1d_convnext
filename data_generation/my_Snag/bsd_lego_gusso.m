function [bsd_out,BSD_tab_out,stpar]=bsd_lego_gusso(addr,ant,runame,tim,freq,mode,modif,modifpost)
	% data access (for Matlab version > 8.1)
	%
	%     [bsd_out,BSD_tab_out,stpar]=bsd_lego(addr,ant,runame,tim,freq,mode,modif,modifpost)
	%
	%    addr      path that contains BSD data master directory without the final dirsep
	%                or structure with all the input parameters as fields
	%    ant       antenna name or BSD table (charged)
	%    runame    run name (ex.: 'O2' for BSD_tab_O2_L.mat, 'C01_O3' for BSD_tab_C01_O3_L
	%    tim       [tini tfin] (mjd) ; if length(tim) = 1 or tini=tfin the whole run
	%              [tini tfin dt0] if the sampling time dt0 of the BSD collection is not 0.1 s
	%    freq      [frini frfin] ; if length(freq) = 1 or frini=frfin the full band with frini (def ortho-band)
	%               (freq is the requested band, band will contain the effective band)
	%    mode      mode  = 0   only the output table
	%                    > 0   creates the output bsd
	%                    = 1   automatic
	%                    = 2   sub-band of concatenated files 
	%                    = 20  sub-band, concatenates files (alternative way)
	%                    = 3   sub-period, many bands - larger band
	%                    = 31  sub-period, many bands - large band
	%                    = 4   inter-bands sub-band
	%   modif      (if present) modification structure: modifies accessed primary files
	%              e.g. adding signals or sources; see bsd_acc_modif
	%              [] no modification
	%   modifpost  (if present) modification after the band extraction
	
	% Snag Version 2.0 - October 2016 remake December 2018
	% Part of Snag toolbox - Signal and Noise for Gravitational Antennas
	% by O.J.Piccinni and S. Frasca - sergio.frasca@roma1.infn.it
	% Department of Physics - Sapienza University - Rome
	
	if length(tim) == 3
		dt0=tim(3);
	else
		dt0=0.1;
	end
	
	if length(tim) > 1
		if tim(1) == tim(2)
			tim=1;
		end
	end
	
	if length(freq) > 1
		if freq(1) == freq(2)
			freq=freq(1);
		end
	end
	
	bsd_out=0;
	icmodif=0;
	
	if exist('modif','var')
		icmodif=1;
	else 
		modif=[];
	end
	
	if ~exist('mode','var')
		mode=1;
	end
	
	if istable(ant)
		tab=ant;
	else
		switch ant
			case 'virgo'
				ach='V';
			case 'ligol'
				ach='L';
			case 'ligoh'
				ach='H';
		end
	
		switch runame
			case 'O1'
				t_ini_run=57277;
				t_fin_run=57369;
			case 'O2'
				t_ini_run=57757;
				t_fin_run=57991;
			case 'O3'
				t_ini_run=58574;
				t_fin_run=58940;
		end
	
		ictim=1;
	
		tab_name=['BSD_tab_' runame '_' ach];
	
		load([addr dirsep 'BSD' dirsep tab_name]);
		eval(['tab=' tab_name ';'])
	end
	
	[BSD_tab_out,stpar]=bsd_subtab(tab,tim,freq,dt0);
	st_t_ini=BSD_tab_out.t_ini;
	st_t_fin=BSD_tab_out.t_fin;
	st_n=(st_t_fin-st_t_ini)*86400/dt0+1;
	
	if isfield(stpar,'Nfft0') % corr 210211
		Nfft0=stpar.Nfft0;
		dt=stpar.dt; 
		band=stpar.band;
	end
	
	if mode == 1 
		if stpar.n_f == 1
			mode=2;
		elseif stpar.n_f == 2
			mode=4;
		elseif stpar.n_f > 2
			mode=3;
		end
		% fprintf(' --- %d mode chosen \n',mode)
	end
	
	if mode == 2 | mode == 4
		freq(2)=freq(2)*(1-eps); % WORKAROUND FOR UPPER LIMIT PROBLEM (for overlapping bands) 19-05-23
		bandin=freq(1:2);
		icm=mode/2;
		ort=bsd_orthoband_gusso(bandin,dt0/icm,Nfft0*icm); 
		stpar.ort=ort;
		freq=ort.bandout;
		bandw=freq(2)-freq(1);
		dt=ort.dt1;
		sbpar.band=ort.bandout;
		sbpar.Nfft0=ort.Nfft0;
		sbpar.Nfft=ort.Nfft1;
		sbpar.dt=ort.dt1;
		sbpar.dfr=ort.dfr1;
		freq=ort.bandout;
		% fprintf(' Ortho-band freq changed:\n ori %f %f  new freq = %f %f  band %f  dt %f\n',bandin,freq,bandw,dt)
			
		sbpar.if1=ort.ifr(1);
		sbpar.if2=ort.ifr(2);
		sbpar.kk=ort.kk;
		stpar.sbpar=sbpar;
		sbpar.tsbase=0;
	end
	
	if mode > 0
		switch mode
			case 2
				if stpar.single_f == 0
					disp(' *** error for mode 2')
					return
				end
				for i = 1:stpar.N
					[in, name]=load_tab_bsd(addr,BSD_tab_out,i,modif);
					cont=cont_gd(in);
					if i == 1
						t0=cont.t0;
					else
						t1=cont.t0;
						jj=diff_mjd(t0,t1)/dt0;
						ndel=mod(jj,sbpar.kk);
						cut=sbpar.kk-ndel;
						y=y_gd(in);
						cont.t0=t1+cut*dt0/86400;
						in=edit_gd(in,'y',y(cut+1:end),'cont',cont);
					end
					sb=bsd_subband_lego(in,sbpar);
					if isfield(cont,'tfstr')
						if isfield(cont.tfstr,'zeros')
							sb=bsd_zeroholes(sb);
						end
					end
					if exist('modifpost','var')
						sb=bsd_acc_modif(sb,modifpost);
					end
					if i == 1
						bsd_out=sb; 
					else
						bsd_out=conc_bsd(bsd_out,sb);
					end
				end
			case 20
				bsd_out=bsd_concsubband(addr,BSD_tab_out,band,modif);
				if ictim == 1
	%                 bsd_out=cut_bsd(bsd_out,tim);
				end
			case 3
				if stpar.single_t == 0
					disp(' *** error for mode 3')
					return
				end
				bsd_out=bsd_largerband(BSD_tab_out,tim(1),tim(2),addr);
			case 31
				if stpar.single_t == 0
					disp(' *** error for mode 31')
					return
				end
				bsd_out1=bsd_large_band(BSD_tab_out,tim(1),tim(2),addr);
				sbpar.band=freq;
				sbpar.tsbase=0;
				sbpar.dt=dx_gd(bsd_out1);
				bsd_out=bsd_subband_lego(bsd_out1,sbpar);
			case 4
				bsd_out=bsd_interband_lego(addr,BSD_tab_out,band,tim,sbpar);
		end
	end
	
	cont=cont_gd(bsd_out);
	[zhole, shole]=bsd_holes(bsd_out);
	cont.tfstr.zeros=shole;
	bsd_out=edit_gd(bsd_out,'cont',cont);
