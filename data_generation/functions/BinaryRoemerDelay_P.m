%Copyright (C) July 2015  Paola Leaci (pleaci@gmail.com)
%Binary Roemer delay computation for both the low- and high-eccentricity cases.

%This program is free software: you can redistribute it and/or modify
%it under the terms of the GNU General Public License as published by
%the Free Software Foundation, either version 3 of the License, or
%(at your option) any later version.

%This program is distributed in the hope that it will be useful,
%but WITHOUT ANY WARRANTY; without even the implied warranty of
%MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%GNU General Public License for more details.

%You should have received a copy of the GNU General Public License
%along with this program.  If not, see <https://www.gnu.org/licenses/>.

function [Rct Rdotc Rddotc] = BinaryRoemerDelay_P(t,sour,radsec,EccAnomOption)

%Example on how running it:
%sour = pulsA_test; tempi = 0:10: 1e6;
%RctLE = BinaryRoemerDelay_P(tempi,sour,0,0);  %Low-ecc case
%sour.ecc = 0.8; RctHE = BinaryRoemerDelay_P(tempi,sour,0,1); %High-ecc case
%figure;plot(tempi, RctLE,'b');hold on;plot(tempi, RctHE,'r'); legend('low ecc','high ecc')

%!ATTENTION! t, ap and P must be anyway in seconds! argp and tp follow radsec below...

% ap = asini/c = semi-major axis projected along the line of sight (s)
% ecc = orbital eccentricity
% arg_p = argument of periapsis passage (rad); it is defined between 0 and 2*pi
%% w = arg_p;
% eta = ecc * sin(arg_p); %First Laplace-Lagrange parameter = EPS1
% K = ecc * cos(arg_p); %Second Laplace-Lagrange parameter = EPS2 
% Om = 2*pi/P = mean orbital angular velocity
% P = orbital period (s)
% tasc = time of ascending node (GPS s)
% tp = time of periapsis passage (GPS s) 
% t = time at which I want to compute Rct (s), after taking into account the Earth’s Roemer delay

if ~exist('t','var')
   disp('Time is required!')
   exit
end

if ~exist('radsec','var')
   disp('Specify if your input is in rad and sec (0) or degrees and mjd (1)')
   exit
end  

if ~exist('sour','var') | (exist('sour','var') && isempty(sour))
   ap =  unifrnd(1.0,5.0); %in lt-s
   period = unifrnd(10*3600.0,24*3600.0); %in s
   ecc = unifrnd(1e-5,0.9);
   arg_p = unifrnd(0.0,2*pi); %in rad
   tref=0;
   tp = tref + unifrnd(- period/2.0, + period/2.0); 


else
   ecc = sour.ecc;
   ap = sour.ap;
   period = sour.P; %in sec

   if radsec ==1 %I am providing in input radians and seconds
      arg_p= sour.argp; %in radians
      if arg_p<0
         disp('arg_p must be between 0 and 2*pi')
         exit      
      end

      if isfield(sour, 'tp')
         tp = sour.tp; %in GPS sec
      end      
      if isfield(sour, 'tasc')
      tasc = sour.tasc; %in GPS sec
      end

   else  
      if sour.argp<0
         disp('arg_p is defined between 0 and 360 degrees')
         exit       
      end
      arg_p=sour.argp*pi/180.0; %longitude of periastron converted in rad

      if isfield(sour, 'tp')
          tp=mjd2gps(sour.tp); %mjd converted in s
      end
      if isfield(sour, 'tasc')
         tasc = mjd2gps(sour.tasc); %in GPS sec
      end
   end

end

if ~exist('EccAnomOption','var')
   EccAnomOption = 0; %It will use by default the small-eccentricity approximation
end

Om = 2*pi/period;


if EccAnomOption == 0  %small-eccentricity approximation
   %disp('ha')
   if isfield(sour,'EPS1') && isfield(sour,'EPS2')
      eta = sour.EPS1; %First Laplace-Lagrange parameter = EPS1
      K = sour.EPS2; %Second Laplace-Lagrange parameter = EPS2
      ecc	= sqrt( K^2 + eta^2 );  %Useful to compute to eventually switch from low- to high-ecc case adding this possibility into this code
      argp 	= mod ( atan2 ( eta, K ), 2*pi );
   else
      K = ecc * cos(arg_p); %Second Laplace-Lagrange parameter = EPS2
      eta = ecc * sin(arg_p); %First Laplace-Lagrange parameter = EPS1
   %tasc = tp - tref - arg_p/Om;
   end
   if ~isfield(sour, 'tasc')
       tasc = tp - arg_p/Om;
   end
   Psi = Om * (t - tasc);
   Rct = ap * ( sin(Psi) + K/2.0 * sin(2*Psi) - eta/2.0 * cos(2*Psi) - 3*eta/2.0 ); %Rct = R(t)/c
   Rdotc = ap * Om * ( cos(Psi) +  K * cos(2*Psi) + eta * sin(2*Psi) ); %time derivative of Rct 
   Rddotc = ap * Om^2 * ( -sin(Psi) -2*K*sin(2*Psi) + 2*eta*cos(2*Psi) ); %time derivative of Rdotc 

else %It will solve the trascendent Kepler's equation (computing the eccentric anomaly with a recursive approach)

   Et = newtonP(t,sour,radsec);
   Rct = ap*(sin(arg_p)*(cos(Et)-ecc)+sqrt(1-ecc^2)*sin(Et)*cos(arg_p)); %=R_c

   Etdot = Om./( 1 - ecc * cos(Et) ); %time derivative of Et   
   Rdotc = ap * Etdot .* ( -sin(arg_p)*sin(Et) + cos(arg_p)*cos(Et)*sqrt(1-ecc.*ecc) ); 

   Etddot = - (Om .* sin(Et) * ecc .* Etdot)/((1 - ecc * cos(Et)).^2); 
   Rddotc = Etdot.^2 * ap .* (-sin(arg_p)*cos(Et) - cos(arg_p)*sin(Et)*sqrt(1-ecc*ecc)) + ap .* Etddot * (-sin(arg_p)*sin(Et) + cos(arg_p)*cos(Et)*sqrt(1-ecc*ecc)); %Rddot/c = acceleration/c

end
