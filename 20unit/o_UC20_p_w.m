clear all;
clc;
%%%%%%%%%%%%%%%%ԭʼ����¼��%%%%%%%%%%%%%%%%%%%%%% 
mpc = o_UC20data;
pump = pump_gen;   %����
Pw=[165 145 120 160 140 120 130 80 35 10 75 85 50 115 125 170 150 195 140 240 140 70 10 80];

load=[700 750 850 950 1000 1100 1150 1200 1300 1400 1450 1500 1400 1300 1200 1050 1000 1100 1200 1400 1300 1100 900 800];
load=2*load;

ge_num=20;   %%�������Ŀ
time_num=24;  %%�������ݼ������Ϊ1Сʱ��
reserve_ratio=0.1;
reserve=reserve_ratio*load;


Pmax=mpc.gencost(:,2);
Pmin=mpc.gencost(:,3);
DT=mpc.gencost(:,4);
a=mpc.gencost(:,5);
b=mpc.gencost(:,6);
c=mpc.gencost(:,7);
CST=mpc.gencost(:,8);
HST=mpc.gencost(:,9);
Tcold=mpc.gencost(:,10);

%% %%%%%%%%%%%%%%%%Ŀ�꺯��%%%%%%%%%%%%%%%%%%%%%
p=sdpvar(ge_num,time_num); %%���������
onoff=binvar(ge_num,2*time_num);  %����״̬����
onoff(:,1:24)=mpc.onoff0;
y=binvar(ge_num,time_num,3);   %������־����
z=binvar(ge_num,time_num,3);   %������������
%% %%%%%%%%%%%%%%ˮ��%%%%%%%%%%%%%%%%%%%%%%
% ���������Ϣ
EU0=pump.gen(:,1);   %��ˮ����ݳ�ֵMWh
ED0=pump.gen(:,2);   %��ˮ����ݳ�ֵ
EUmin=pump.gen(:,3); %��ˮ�������Сֵ
EUmax=pump.gen(:,4); %��ˮ��������ֵ
EDmin=pump.gen(:,5); %��ˮ�������Сֵ
EDmax=pump.gen(:,6); %��ˮ��������ֵ
PPmax=pump.gen(:,7); %��ˮ���ֵ
PGmax=pump.gen(:,8); %�������ֵ
Tau1=pump.gen(:,9);  %����Ч��
Tau2=pump.gen(:,10); %��ˮЧ��
M=10000;

% ��������
EU=sdpvar(1,time_num);    %��ˮ�����
ED=sdpvar(1,time_num);    %��ˮ�����
Ph=sdpvar(1,time_num);    %��ˮ���繦��
Phmin=sdpvar(1,time_num);
Phmax=sdpvar(1,time_num);
wz=binvar(2,time_num);

%% %%%%%%%%%%%%%%%%Ŀ�꺯��%%%%%%%%%%%%%%%%%%%%%
F1=0;
for i=1:ge_num
    for k=1:time_num
        t=k+24;
            F1=F1+onoff(i,t)*a(i) + b(i)*p(i,k) + c(i)*p(i,k)^2+z(i,k,1)*HST(i)+z(i,k,2)*CST(i);
    end
end
       
%% %%%%%%%%%%%%%%%Լ������%%%%%%%%%%%%%%%%%%%%%%
Constraint=[ ];

% ����
for i=1:ge_num
    for k=1:time_num
        t=k+24;  
           %�������Լ��
            Constraint=[Constraint, onoff(i,t)*Pmin(i) <= p(i,k) <=onoff(i,t)*Pmax(i)];
            %
            Constraint=[Constraint, z(i,k,1)+z(i,k,2)-z(i,k,3) <= onoff(i,t)-onoff(i,t-1) <= z(i,k,1)+z(i,k,2)];
            
            Constraint =[Constraint, z(i,k,1)<=sum(onoff(i,t-(Tcold(i)+DT(i)+1):t-1))<=(z(i,k,1)+z(i,k,3))*(Tcold(i)+DT(i)+1)];
            
            Constraint=[Constraint, z(i,k,1)+z(i,k,2)+z(i,k,3) == 1, y(i,k,1)+y(i,k,2)+y(i,k,3) == 1];
            
            Constraint = [Constraint, onoff(i,t)-onoff(i,t-1) == y(i,k,1)-y(i,k,2), ...
                y(i,k,2)*DT(i)<=sum(onoff(i,t-DT(i):t-1))<=(y(i,k,2)+y(i,k,3))*DT(i)];
    end
end

for i=1:10
    for k=1:time_num
        t=k+24;
        Constraint=[Constraint, onoff(i+10,t) <= onoff(i,t)];
    end
end


% ����
Constraint=[Constraint, EU(1,1) == EU0, ED(1,1) == ED0];   %���ݸ���ֵ
for k=2:time_num
    % ����ˮ�����
    Constraint=[Constraint, EUmin <= EU(1,k) <= EUmax];
    Constraint=[Constraint, EDmin <= ED(1,k) <= EDmax];
end

for k=1:time_num-1
    Constraint=[Constraint, -wz(2,k)*M <= Ph(1,k) <= wz(1,k)*M];
    Constraint=[Constraint, wz(1,k)+wz(2,k) == 1];
    %Ph>=0 ����
    Constraint=[Constraint, -wz(2,k)*M <= EU(1,k+1)-EU(1,k)+Ph(1,k)/Tau1 <= wz(2,k)*M];
    Constraint=[Constraint, -wz(2,k)*M <= ED(1,k+1)-ED(1,k)-Ph(1,k)/Tau1 <= wz(2,k)*M];
    %Ph<=0 ��ˮ
    Constraint=[Constraint, -wz(1,k)*M <= EU(1,k+1)-EU(1,k)+Ph(1,k)*Tau2 <= wz(1,k)*M];
    Constraint=[Constraint, -wz(1,k)*M <= ED(1,k+1)-ED(1,k)-Ph(1,k)*Tau2 <= wz(1,k)*M];
end

for k=1:time_num
    % ����繦�ʱ仯
    Constraint=[Constraint, Phmin(1,k) <= Ph(1,k) <= Phmax(1,k)];
    Constraint=[Constraint, Phmin(1,k) == max(-PPmax,-ED(1,k)/Tau2), Phmax(1,k) == min(PGmax,EU(1,k)*Tau1)];
end

%% ����ƽ��Լ��
for k=1:time_num
    Constraint=[Constraint, sum(p(:,k))+Ph(1,k)+Pw(1,k) == load(k)];
end

%% ��ת����Լ��
for k=1:time_num
    t=k+24;
    Constraint=[Constraint, sum(Pmax'*onoff(:,t))+Ph(1,k)+Pw(1,k)-load(k) >= reserve(k)];
end

%% %%%%%%%%%%%%�Ż���������%%%%%%%%%%%%%%
ops = sdpsettings('solver','gurobi');
sol = solvesdp(Constraint,F1,ops);
