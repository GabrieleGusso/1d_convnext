function out=bsd_orthoband_gusso(bandin,dt0,Nfft0)
	% search orthoband for given band, sampling time and Nfft
	%
	%   bandin   input band
	%   dt0      base sampling time
	%   Nfft0    length of the ffts (number of frequencies)
	
	% Snag Version 2.0 - December 2018
	% Part of Snag toolbox - Signal and Noise for Gravitational Antennas
	% by O.J.Piccinni and S. Frasca - sergio.frasca@roma1.infn.it
	% Department of Physics - Sapienza University - Rome
	
	Nband=diff(floor(bandin*dt0))+1;
	out.Nband=Nband;
	base_band=1/dt0;
	
	if Nband == 1
		out.obtyp=1;
		% disp('Short band')
	elseif Nband == 2 && bandin(2)-bandin(1) < base_band
		out.obtyp=2;
		disp(' *** Two basic bands sub-band')
		disp('Create a 2 adjacent band BSD')
		disp('Use Inter-band (mode 4)')
		out.error=' *** Two basic bands sub-band; use inter-band';
		disp('The orthoband parameter for the couples are produced')
		return
	else
		out.obtyp=0;
		disp('Multi-band --- FULL BAND - (there is no orthoband)')
		out.error=' *** Use Full Band mode (mode 3 only for short time)';
		return
	end
	
	bandtot=1/dt0;
	Nob=200;
	eps0=bandtot*0.001/Nfft0;
	
	fr0=floor(bandin(1)/bandtot)*bandtot;
	band=bandin-fr0;
	
	for kk = Nob:-1:1
		wob=bandtot/kk;
		i1=floor(band(1)/wob+eps0);
		b(1)=i1*wob;
		b(2)=b(1)+wob;
		if b(1) <= band(1)+eps0 && b(2) >= band(2)-eps0
			jj=i1+1;
			break
		end
	end
	
	out.bandin=bandin;
	out.wbandin=round(diff(bandin),12);
	out.dt0=dt0;
	out.Nfft0=Nfft0;
	out.fr0=fr0;
	out.Nob=Nob;
	
	out.kk=kk;
	out.jj=jj;
	out.wob=wob;
	out.bandout=b+fr0;
	out.wbandout=round(diff(b),12);
	out.eps=eps0;
	
	nwob=ceil(Nfft0*wob*dt0);
	
	out.nwob=nwob;
	out.Nfft1=nwob*kk;
	out.dfr1=1/(dt0*out.Nfft1);
	out.bandout_d=round(round(b/out.dfr1)*out.dfr1,12);
	out.ifr=round(b/out.dfr1)+[1 0];
	% out.dt1=round(1/out.wbandout,13);
	out.dt1=dt0*kk; 
