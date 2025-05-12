function [out Hp Hc sig ph0]=bsd_softinj_re_mod_isobin(in,sour,A,DT,radsec,EccAnomOption)
% Software injection either iso or bin
%
%    out=bsd_softinj_re(in,sour,A)
% 
%   in              input bsd
%   sour            source structure
%   A               amplitude
%   DT              difference between the first time samples of LHO & LLO (i.e., 2 IFOs with comparable sensitivity)
%   radsec          1 if you are taking in input radians and seconds, 0 otherwise; USEFUL only for binaries; for iso specify 0
%   EccAnomOption   0 (=default) small-eccentricity approximation (from 0 up to 0.1); 1 high-ecc approx; 'none' isolated case


% Snag Version 2.0 - February 2019
% Part of Snag toolbox - Signal and Noise for Gravitational Antennas
% by O.J.Piccinni, S. Frasca - sergio.frasca@roma1.infn.it
% Department of Physics - Sapienza University - Rome

%Paola Leaci for binary stuff, 2016-2024

if ~exist('radsec','var')
    disp('specify if you want to work in rad and sec (0) or degrees and mjd (1)')
    %radsec = 0;  %as I am NOT working with the default w in radians and P in sec, but rather w in degrees, P in days and tp in mjd
    exit
 end 
 if ~exist('EccAnomOption','var')
    EccAnomOption = 'none'; % isolated case
end
if ~exist('DT','var')
    DT=0;
end
Tsid=86164.09053083288; % at epoch2000

frsid=1/Tsid;
inisid1=0;

N=n_gd(in);         % get dimension of the gd
dt=dx_gd(in);       % get sampling "time"
cont=cont_gd(in);   % get control variable
y=y_gd(in);         % get ordinate
t0=cont.t0;
inifr=cont.inifr;
ant1=cont.ant;
eval(['ant=' ant1 ';'])

sour=new_posfr(sour,t0);    % recomputes position and frequency
fr=[sour.f0,sour.df0,sour.ddf0];
f0=sour.f0;
if exist('sdpar','var')
    if length(sdpar) == 1
        sdpar(2)=0;
    end
    fr(2)=sdpar(1);
    fr(3)=sdpar(2);
else
    sdpar(1)=fr(2);
    sdpar(2)=fr(3);
end

% FFT on gds
obs=check_nonzero(in,1);                % checks for nonzero intervals
ph0=mod((0:N-1)*dt*(f0-inifr),1)*2*pi;
in=edit_gd(in,'y',exp(1j*ph0).*obs');

VPstr=extr_velpos_gd(in);               % extracts velocity and position from a BSD gd
[p0, v0]=interp_VP(VPstr,sour);         % position and velocity interpolation

out = vfs_subhet_pos_isobin(sour,in,fr,p0,DT,radsec,-1,EccAnomOption); % application of sub-heterodyne

if sum(abs(sdpar)) > 0
    gdpar=[dt,N];

    sd=vfs_spindown(gdpar,sdpar,1);

    out=vfs_subhet(out,-sd);
end

einst=einst_effect(t0:dt/86400:t0+(N-0.5)*dt/86400);    % Einstein effect as frequency variation for Earth orbit eccentricity
out=vfs_subhet(out,(einst-1)*fr(1));                    % application of sub-heterodyne
%fprintf('No Einstein delay!\n');

sig0=y_gd(out); % get ordinate
sig=sig0*0;

% 5-vect

nsid=10000;
stsub=gmst(t0)+dt*(0:N-1)*(86400/Tsid)/3600; % running Greenwich mean sidereal time 
isub=mod(round(stsub*(nsid-1)/24),nsid-1)+1; % time indexes at which the sidereal response is computed

[~, ~ , ~, ~, sid1 sid2]=check_ps_lf(sour,ant,nsid,0); % computation of the sidereal response (computes the low frequencies for the 4 basic polarizations in one sidereal day)
gl0=sid1(isub).*sig0.';
gl45=sid2(isub).*sig0.';
Hp=sqrt(1/(1+sour.eta^2))*(cos(2*sour.psi/180*pi)-1j*sour.eta*sin(2*sour.psi/180*pi)); % estimated + complex amplitude (Eq.2 of ref.1)
Hc=sqrt(1/(1+sour.eta^2))*(sin(2*sour.psi/180*pi)+1j*sour.eta*cos(2*sour.psi/180*pi));
sig=Hp*gl0+Hc*gl45;

sig=sig.';

out=edit_gd(out,'y',A*sig+y); % A*sig+y = segnale + rumore; A*sig = solo puro segnale; y = solo rumore
