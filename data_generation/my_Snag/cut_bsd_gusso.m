function [bsd_o,i1,i2]=cut_bsd_gusso(bsd_i,tt,enl,forcesec)
% cuts a bsd 
%
%    bsd_i     input bsd
%    tt        [tin tfi] time interval (mjd or seconds from beginning)
%    enl       > 0 -> enlarge if needed (def 0)
%    forcesec  > 0 forces input in s (def 0)

% Version 2.0 - October 2016
% Part of Snag toolbox - Signal and Noise for Gravitational Antennas
% by O.J.Piccinni and S. Frasca - sergio.frasca@roma1.infn.it
% Department of Physics - Sapienza University - Rome

if ~exist('enl','var')
    enl=0;
end
if ~exist('forcesec','var')
    forcesec=0;
end
dt=dx_gd(bsd_i);
n=n_gd(bsd_i);
y=y_gd(bsd_i);
cont=cont_gd(bsd_i);
icz=0;
if isfield(cont,'tfstr')
    if isfield(cont.tfstr,'zeros')
        zeros1=cont.tfstr.zeros;
        icz=1;
    end
    cont.tfstr=[];
end
t0=cont.t0;
v_eq=cont.v_eq;
p_eq=cont.p_eq;
t_dop=adds2mjd(t0,v_eq(:,4));

oper=struct();
oper.op='cut_bsd';
if isfield(cont,'oper')
    oper.oper=cont.oper;
end
oper.tt=tt;

cont.oper=oper;

icmjd=0;
tlim1=50000;
tlim2=90000;

if tt(1) > tlim1 &  tt(1) < tlim2
    icmjd=1;
end
if forcesec > 0
    icmjd=0;
end

switch icmjd
    case 0
        sec0=tt(1);
        secs=tt(2)-tt(1);
    case 1
        secs=diff_mjd(tt(1),tt(2));
        sec0=diff_mjd(t0,tt(1));
end

i1=round(sec0/dt)+1;
i2=round((sec0+secs)/dt);
if i1 < 1  % workaround for possible strange problem 25-1-2019
    i1=1;
end
if i2 > n
    i2=n;
end
ts1=(i1-1)*dt;
ts2=(i2-1)*dt;

y=y(i1:i2);
t0_new=adds2mjd(t0,ts1);
tfin=adds2mjd(t0_new,secs);
ii=find(t_dop >= t0_new & t_dop <= tfin);
v_eq=v_eq(ii,:);
v_eq(:,4)=v_eq(:,4)-ts1;
p_eq=p_eq(ii,:);
p_eq(:,4)=p_eq(:,4)-ts1;

cont.t0=t0_new;
cont.v_eq=v_eq;
cont.p_eq=p_eq;
if icz == 1
    ii=find(zeros1(:,2) > sec0);
    zeros1=zeros1(ii,:);
    ii=find(zeros1(:,1) < sec0+secs);
    zeros1=zeros1(ii,:);
    cont.tfstr.zeros=zeros1-diff_mjd(t0,t0_new);
end

bsd_o=edit_gd(bsd_i,'y',y,'cont',cont);

if enl > 0
    if icmjd == 0
        disp(' *** Enlargement possible only with mjd limits')
        return
    end
    if tt(1) < t0-eps
        nsam1=round(diff_mjd(tt(1),t0)/dt);
        ts1=nsam1*dt;
        cont.t0=adds2mjd(t0,-ts1);
        y=y_gd(bsd_o);
        n0=n_gd(bsd_o);
        y(nsam1+1:nsam1+n0)=y;
        y(1:nsam1)=0;
        if icz == 1
            zeros1=cont.tfstr.zeros;
            [nz,~]=size(zeros1);
            zeros1(2:nz+1,:)=zeros1+ts1;
            zeros1(1,1)=0;
            zeros1(1,2)=nsam1*dt;
            cont.tfstr.zeros=zeros1;
        end
        
        v_eq=v_eq(ii,:);
        v_eq(:,4)=v_eq(:,4)+ts1;
        p_eq=p_eq(ii,:);
        p_eq(:,4)=p_eq(:,4)+ts1;

        cont.v_eq=v_eq;
        cont.p_eq=p_eq;
        
        bsd_o=edit_gd(bsd_o,'y',y,'cont',cont);
    end
    tfin=adds2mjd(cont.t0,(n_gd(bsd_o)-1)*dt);
    if tt(2) > tfin+eps
        nsam2=round(diff_mjd(tfin,tt(2))/dt); 
        y=y_gd(bsd_o);
        n0=n_gd(bsd_o);
        y(n0+1:n0+nsam2)=0;
        if icz == 1
            zeros1=cont.tfstr.zeros;
            [nz,~]=size(zeros1);
            zeros1(nz+1,1)=n0*dt;
            zeros1(nz+1,2)=(n0+nsam2-1)*dt;
            cont.tfstr.zeros=zeros1;
        end
        bsd_o=edit_gd(bsd_o,'y',y,'cont',cont);
    end
end
