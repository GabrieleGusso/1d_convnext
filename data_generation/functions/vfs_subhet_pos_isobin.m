function [out, BinaryDelays, corr]=vfs_subhet_pos_isobin(sour,in,fr,pos,DT,radsec,exp_sign,EccAnomOption)
% application of sub-heterodyne
%
%
%   in               input data gd
%   fr               frequency (fixed, varying or spin-down [f df ddf ...])
%   pos              position (normalized by c)
%   DT               difference between the first time samples of LHO & LLO (i.e., 2 IFOs with comparable sensitivity)
%   radsec           1 if you are taking in input radians and seconds, 0 otherwise; USEFUL only for binaries; for iso specify 0
%   EccAnomOption    0 (=default) small-eccentricity approximation (from 0 up to 0.1); 1 high-ecc approx; 'none' isolated case

%Binary functions Paola Leaci; 2016-2024

% Version 2.0 - October 2015
% Part of Snag toolbox - Signal and Noise for Gravitational Antennas
% by Sergio Frasca - sergio.frasca@roma1.infn.it
% Department of Physics - Sapienza University - Rome

if ~exist('exp_sign','var')
    exp_sign = +1; %+1 demodulo; -1 modulo, useful for injections
end
if ~exist('radsec','var')
   disp('specify if you want to work in rad and sec (0) or degrees and mjd (1)')
   exit
end 
if ~exist('DT','var')
    DT=0;
end

dt=dx_gd(in);
t=x_gd(in);
n=n_gd(in);
cont = cont_gd(in);
nfr=length(fr);


switch nfr
    case 1
        ph=pos*fr*2*pi;
    case n
        ph=pos.*fr*2*pi;
    otherwise
        fr0=fr(1);
        sd = fr(2);
        %phi_sd = 2*pi*sd*(t).^2/2;
        fr=vfs_spindown([dt,n],fr(2:nfr),1,DT)+fr0;
        
        tempi = mjd2gps(cont.t0) + (0 : n-1)*dt + pos.';
        
        if EccAnomOption == 'none' | ~isfield(sour,'binary') %isolated case
            BinaryDelays = 0.0;
        else
            [Rct u]= BinaryRoemerDelay_P(tempi,sour,radsec,EccAnomOption);
            Binary_Roemer = - Rct; %Binary Roemer delay 

            EinstShapiro =1; %It can be a CLA (0 or 1)
            if ~EinstShapiro == 0 %it will correct for those effects 
                G = 6.67259e-11; %N m2 /kg
                c = 299792458; %m/s
                Msun=1.988e30;
                gamma = sour.gamma; %Amplitude of binary Einstein delay
                M2 = sour.m2* Msun; %m2=companion mass in solar masses, which I need to transform in kg!!!
                sini = sour.sini;
                
                DeltaEinstein = gamma * sin(u); %Eq. 78 of MNRAS 372, 1549-1574 (2006)
                r = G*M2/c^3;
                %Correction for the post-Keplerian Binary Shapiro delay
                DeltaShapiro = -2*r*log( 1-sour.ecc*cos(u)-sini*( sin(sour.argp)*(cos(u)-sour.ecc) + sqrt(1-sour.ecc^2)*cos(sour.argp)*sin(u) ) ); %Eq. 79 of MNRAS 372, 1549-1574 (2006)
            
                DeltaEDeltaS = DeltaEinstein + DeltaShapiro;
            else %if EinstShapiro == 0, it will no correct for those effects
                DeltaEDeltaS = 0;
            end
            
            BinaryDelays = Binary_Roemer + DeltaEDeltaS;

        end

        ph_binary=(BinaryDelays.*fr*2*pi).';
        
        ph=(2*pi*fr.*(pos.')).' + ph_binary;
        ph = exp_sign * ph; 

end

corr=exp(-1j*(ph));
out=y_gd(in).*corr(:);
out=edit_gd(in,'y',out);
