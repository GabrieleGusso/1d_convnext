%Copyright (C) 2015, 2024  Paola Leaci (pleaci@gmail.com)
%newtonP() solves Kepler's equation via the Newton–Raphson method

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


function E_old = newtonP(tt,sour,radsec)

if ~exist('radsec','var')
    disp('Specify if your input is in rad and sec (0) or degrees and mjd (1)')
    exit
end  

ecc = sour.ecc; %orbital eccentricity
ap = sour.ap;  %ap = asini/c = semi-major axis projected along the line of sight (s)
period = sour.P; %orbital period in sec

if radsec ==1 %I am providing in input radians and seconds
    arg_p= sour.argp; %argument of periapsis passage in radians
    
    if isfield(sour, 'tp')
        tp = sour.tp; % time of periapsis passage (GPS s) 
    end
     
    if isfield(sour, 'tasc')
     tasc = sour.tasc; %in GPS sec
    end

else  
    arg_p=sour.argp*pi/180.0; %longitude of periastron converted in rad
    %period = sour.P*86400.0; % orbital period converted in seconds
    
    if isfield(sour, 'tp')
        tp=mjd2gps(sour.tp); %mjd converted in s
    end
    if isfield(sour, 'tasc')
       tasc = mjd2gps(sour.tasc); %in GPS sec
    end
end

Omega = 2*pi/period;

if ~isfield(sour, 'tp')
    fe=sqrt((1-ecc)/(1+ecc)); % fractional eccentric anomaly (fe) based on the orbital eccentricity ecc
    uasc=2*mod(atan2(fe*tan(arg_p/2.0),1),2*pi); % true anomaly (uasc) at the ascending node
    tp = tasc + (uasc-ecc*sin(uasc))/Om; 
end

ot=Omega.*(tt-tp); %Kepler's equation
E_old=ot;
%E_old = zeros(1, length(tt))+1;
tol=100;
i=0;
numIter = 60; %number of iterations
t0=cputime;
while(tol>1e-10 && i<numIter)
    i=i+1;
    E_new=E_old-(E_old-ecc*sin(E_old)-ot)./(1.0-ecc*cos(E_old));
    tol=max(abs(E_new-E_old));
    %disp(tol)
    E_old=E_new;
end
t1=cputime;
fprintf('eccen: %f iteration: %d tol: %E time: %f s\n',ecc,i,tol, t1-t0)
% m=E_old-ecc*sin(E_old);
% er=max(abs(m-ot));
% disp(er)
% %plot(t,E_old,t,ot)
% plot(tt,E_old)
