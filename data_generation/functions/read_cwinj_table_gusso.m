function pulsar=read_cwinj_table_gusso(dataFile,kpuls)
    % Creates a cw source structure from FakeBinPulsarParameters list
    %
    %   dataFile   cw injection fake pulsar parameters (load FakeBinPulsarParameters.dat)
    %   kpuls      number of the pulsar
    
    % April 2024
    % by G. Gusso - gusso@roma1.infn.it
    % Department of Physics - Sapienza University - Rome
    
    % injtab=removevars(readtable(dataFile),{'Var3'});
    injtab=readtable(dataFile);
    idx = 1+15*(kpuls-1) : 1 : 15*kpuls;

    varData=injtab.varData(idx);

    Alpha=varData(2);           % right ascention [rad] must be converted to pulsar.a [deg] and after pulsar.ecl [deg]
    Delta=varData(3);           % declination [rad] must be converted to pulsar.b [deg] and after pulsar.ecl [deg]
    h0=varData(4);              % it's h0 and must be converted to H0 wich is defined as pulsar.h = h0*sqrt(1+6*cosi^2+cosi^4)/2
    cosi=varData(5);            % cosine of the inclination angle of the NS rotational axis to the line of sight
    psi=varData(6);             % mean orbital phase pulsar.psi = psi
    %phi0=varData(7);            % fase iniziale della cw nel sistema di riferimento della stella
    Freq=varData(8);            % emitted CW frequency [Hz] pulsar.f0 = Freq
    orbitasini=varData(9);      % orbital projected semi-major axis [light-seconds] pulsar.ap = orbitasini [light-seconds]
    orbitEcc=varData(10);       % orbital eccentricity pulsar.ecc = orbitEcc
    orbitTp=varData(11);        % SSB time of periapsis [seconds] pulsar.tp = orbitTp [seconds]
    orbitPeriod=varData(12);    % orbital period [seconds] pulsar.P = orbitPeriod [seconds]
    orbitArgp=varData(13);      % argument of periapsis [rad] pulsar.argp = orbitArgp [rad]
    f1dot=varData(14);          % spindown [Hz^2] pulsar.df0 = f1dot [Hz^2]
    refTime=varData(15);        % frequency injection time [seconds] must be converted to pulsar.fepoch = gps2mjd(refTime) [mjd]

    % PULSAR FUNDAMENTAL PARAMETERS
    pulsar.name=['pulsar_' num2str(kpuls)];               % name
    pulsar.a=Alpha*180/pi;                                % right ascention [deg]
    pulsar.d=Delta*180/pi;                                % declination [deg]
    pulsar.v_a=0;                                         % right ascention variation [milliarcsec per year]
    pulsar.v_d=0;                                         % declination variation [milliarcsec per year]
    pulsar.pepoch=gps2mjd(refTime);                       % epoca misura sperimentale del periodo orbitale [mjd]
    pulsar.f0=Freq;                                       % emitted CW frequency [Hz]
    pulsar.df0=f1dot;                                     % spindown [Hz^2]
    pulsar.ddf0=0;                                        % spindown variation [Hz^3]
    pulsar.fepoch=pulsar.pepoch;                            % epoca della misura sperimentale della frequenza [mjd]
    [ao,do]=astro_coord('equ','ecl',pulsar.a,pulsar.d);
    pulsar.ecl=[ao,do];                                   % eclittical coordinates: longitude and latitude [deg]

    % PULSAR 
    pulsar.t00=51544;                         % time from 1st january 2000 [mjd]
    pulsar.eps=1;                             % ellipticity, which is a measure of the NS deformation defined in Eq. 8 of Singhal et al. 2019
    pulsar.eta=-2*cosi/(1+cosi^2);            % ratio of the polarization ellipse semi-minor to semi-major axis and indicates the GW degree of polarization, which depends on i, the angle between the rotational axis of the NS with respect to the line of sight pulsar.eta = -2*cosi/(1+cosi^2) defined in Eq. 2 of Singhal et al. 2019
    pulsar.psi=psi*180/pi;                    % polarisation angle, which is defined as the direction of the major axis with respect to the celestial parallel of the source (measured counter-clock- wise) defined in Eq. 4 of Singhal et al. 2019
    pulsar.h=h0*sqrt(1+6*cosi^2+cosi^4)/2;    % it's H0 as defined in Eq. 49 of of Leaci et al. 2017
    pulsar.snr=1;                             % signal-noise ratio

    % BINARY PARAMETERS
    pulsar.P    = orbitPeriod;        %% orbital period [seconds]
    pulsar.ap   = orbitasini;         %% orbital projected semi-major axis [light-seconds]
    pulsar.tp   = orbitTp;            %% SSB time of periapsis [seconds]
    pulsar.ecc  = orbitEcc;           %% orbital eccentricity
    pulsar.argp = orbitArgp;          %% argument of periapsis [rad]

    % BINARY CORRECTIONS
    pulsar.binary= 'no'; % is binary?
    pulsar.Pbdot= 0;      % binary period variation
    pulsar.omdot= 0;      % binary angular velocity variation
    pulsar.m2= 0;         % companion mass [solar masses]
    pulsar.gamma= 0;      % amplitude of binary Einstein delay
    pulsar.sini= 0;       % usato nella correzione dello Shapiro-delay
