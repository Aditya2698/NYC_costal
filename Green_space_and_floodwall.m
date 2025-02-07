% with carbon emissions costs (accumulated and SCC increasing in time)
% actions are green infra (inland) and dike2
% rewards as penalties due to flooding corresponding to each state (level
% of water)

% with a seawall of height 1m

% calculation of accumulated scc
load('scc.mat');
horizon1 = 40;
extra_years = 20; % for terminal costs calculation
horizon = horizon1 + extra_years;
scc(41:60) = scc(40);
discounted_sum_scc = zeros(1, horizon);
discount_factor = 0.97;
for i = 1:horizon
    scc_i = scc(i:end);
    discount = 1;
    for j = 1:length(scc_i)
        discounted_sum_scc(i) = discounted_sum_scc(i) + discount*scc_i(j);
        discount = discount*discount_factor;           
    end
end

% rewards model for the input file
% rewards are negative since they represent costs
n_systems = 4;
% s = 0.0085;% slope of city (without seawall)
% s = 0.0080;% slope of city (with seawall of height 1m)
seawall_h = 1.5;
s = 0.0077;% slope of city (with seawall of height 1.5m)
unit_l = 1;%m
%costs
%cost of construction of each dike
gamma = 0.97;
Dike_Height = 1.5;%m
cc_dike = -Dike_Construction_Cost1(Dike_Height); %$/m length along shore
% cm_dike = 0.01*cc_dike; %$/m
cm_dike = -100;%$/m length along shore
f_damage = 0.07;% 20 billions
val_z = 15.3*1e6;
cf = -f_damage*val_z;
vol_z = (8.50*(1/s)*8.50)*0.5;

[cc_carbon_dike, ] = - DikeConstructionMaterialGHGCost1(Dike_Height,1);
[cm_carbon_dike, ] = - AnnualMaintenanceGHGCost(1);
% cost of construction of NBS

cg = -25;%$/m2
b_green = 1.7;
t_green = 3.2;
l_green = sqrt(((t_green - b_green)/s)^2 + (t_green - b_green)^2);%m
cc_green = cg*l_green; %$/m

% cost of maintenance of NBS
cm_nbs_l = -2.7; %$/m2
cm_green = cm_nbs_l*l_green; %$/m

% wave attenuation of NBS
wa_nbs_m = 0.1/100; % % per m length submerged: NOT USED

% carbon costs:

% % carbon costs for construction
carbon_green_c = 1.6/1000; %ton CO2 per m2
cc_carbon_green = -(carbon_green_c)*l_green*unit_l;%$/m
% % carbon costs for maintenance
carbon_green_m = 0.09/1000; %ton CO2 per m2
cm_carbon_green = -carbon_green_m*l_green*unit_l;%$/m
% % carbon uptake by green infra
carbon_up_green = 0.17/1000; %ton CO2 per m2
cu_carbon_green = carbon_up_green*l_green*unit_l;%$/m

% action 1- do nothing
R_dn = cell(1,n_systems); 
% rewards for system A under do-nothing action
% risk
slr_states = 77;
surge_states = 72;
n_states = slr_states*surge_states;
vol_f = zeros(1, n_states);
rewards1_dn = zeros(1, n_states);

for i = 1:n_states
       [slr1,surge1] = components(i);
        slrh1 = values_slr(slr1);
        surgeh1 = values_surge(surge1);
        totalh1 = (slrh1 + surgeh1)/100;%m
        if totalh1 <= seawall_h
            totalh1 = 0;
        else
            totalh1 = totalh1 - seawall_h;
        end
        area1 = 0.5*(1/s)*totalh1^2;

        vol_f(i) = (area1 - 0);%area in m^2
        c_flood = cf*vol_f(i)/vol_z;
        R_flood1(i) = c_flood; 
end
%risk carbon
R_flood_carbon1 = zeros(horizon, n_states);
for t = 1:horizon
    c_carbon_flood1 = zeros(1, length(n_states));
for i = 1:n_states
        [~, c_carbon_flood1(i)] = FloodDamageGHGCost(R_flood1(i), discounted_sum_scc(t));      
end
R_flood_carbon1(t,:) = c_carbon_flood1; 
end
% total
rewards1_dn = cell(1, horizon);
for t = 1:horizon
    rewards1_dn{1,t} = (R_flood1 + R_flood_carbon1(t,:))*gamma;
end
R_dn{1,1} = rewards1_dn;


% rewards for system B under do-nothing action
% risk

slr_states = 77;
surge_states = 72;
n_states = slr_states*surge_states;
R_flood2 = zeros(1, n_states);

for i = 1:n_states   
        [slr1,surge1] = components(i);
        slrh1 = values_slr(slr1);
        surgeh1 = values_surge(surge1);
        totalh1 = (slrh1 + surgeh1)/100;
        if totalh1 <= seawall_h
            totalh1 = 0;
            area1 = 0;
        elseif totalh1 > seawall_h && totalh1 <= b_green
             f_damage = 1.0;
             totalh1 = totalh1 - seawall_h;
             area1 = 0.5*f_damage*(1/s)*totalh1^2;
        elseif (totalh1> b_green) && (totalh1 <= t_green)
            f_damage1 = 1.00;
            a1 = (0.5*(1/s)*(b_green - seawall_h)^2);
            f_damage2 = 0.80;
            a2 = (0.5*(1/s)*(totalh1-seawall_h)^2 - a1);
            area1 = a1*f_damage1+a2*f_damage2;
        elseif totalh1 > t_green
            f_damage1 = 1.00;
            a1 = (0.5*(1/s)*(b_green-seawall_h)^2);
            f_damage2 = 0.80;
            a2 = (0.5*(1/s)*(t_green-seawall_h)^2 - a1);
            a3 = (0.5*(1/s)*(totalh1-seawall_h)^2 - a1 - a2);
            area1 = a1*f_damage1+a2*f_damage2+a3*f_damage1; 
        end
        vol_f(i) = (area1 - 0);
        c_flood = cf*vol_f(i)/vol_z;
        R_flood2(i) = c_flood;
end
%risk carbon
R_flood_carbon2 = zeros(40, n_states);
for t = 1:horizon
    c_carbon_flood2 = zeros(1, length(n_states));
for i = 1:n_states
        [~, c_carbon_flood2(i)] = FloodDamageGHGCost(R_flood2(i), discounted_sum_scc(t));      
end
R_flood_carbon2(t,:) = c_carbon_flood2; 
end

%maintenance
R_m = cm_green * ones(1, n_states);
R_mc = zeros(40, n_states);
for t = 1:horizon
    Rmc(t) = cm_carbon_green*discounted_sum_scc(t);
    R_mc(t,:) = ones(1, n_states)*Rmc(t);
end

%carbon uptake
R_up = zeros(40, n_states);
for t = 1:horizon
    Rup(t) = cu_carbon_green*discounted_sum_scc(t);
    R_up(t,:) = ones(1, n_states)*Rup(t);
end

%total
rewards2_dn = cell(1, horizon);
for t = 1:horizon
    rewards2_dn{1,t} = (R_flood2 + R_flood_carbon2(t,:))*gamma + R_m + R_mc(t,:) + R_up(t,:);
end
R_dn{1,2} = rewards2_dn;

% rewards for system C under do-nothing action

slr_states = 77;
surge_states = 72;
n_states = slr_states*surge_states;
b = 3.40;%m
t = 4.90;%m
gamma = 0.97;
R_flood3 = zeros(1, n_states);

for i = 1:n_states
    
        [slr1,surge1] = components(i);
        slrh1 = values_slr(slr1);
        surgeh1 = values_surge(surge1);
        totalh1 = (slrh1 + surgeh1)/100;
        if totalh1 <= seawall_h
            totalh1 = 0;
            area1 = 0;
        elseif (totalh1> b) && (totalh1 <= t)
            area1 = 0.5*(b-seawall_h)*(b-seawall_h)*(1/s) + (totalh1 - b)*(b-seawall_h)*(1/s);
        else
            area1 = 0.5*(1/s)*(totalh1-seawall_h)^2;
        end
        
        vol_f(i) = (area1 - 0);%area in m^2
        c_flood = cf*vol_f(i)/vol_z;
        R_flood3(i) = c_flood;
end
%risk carbon
R_flood_carbon3 = zeros(40, n_states);
for t = 1:horizon
    c_carbon_flood3 = zeros(1, length(n_states));
for i = 1:n_states
        [~, c_carbon_flood3(i)] = FloodDamageGHGCost(R_flood3(i), discounted_sum_scc(t));      
end
R_flood_carbon3(t,:) = c_carbon_flood3; 
end
%maintenance
R_mc = zeros(40, n_states);
R_m = cm_dike * ones(1, n_states);
for t = 1:horizon
    Rmc(t) = cm_carbon_dike*discounted_sum_scc(t);
    R_mc(t,:) = ones(1, n_states)*Rmc(t);
end

%total
rewards3_dn = cell(1, 40);
for t = 1:horizon
    rewards3_dn{1,t} = (R_flood3 + R_flood_carbon3(t,:))*gamma + R_m + R_mc(t,:);
end
R_dn{1,3} = rewards3_dn;

% rewards for system D under do-nothing action

slr_states = 77;
surge_states = 72;
n_states = slr_states*surge_states;
b2 = 3.40;%m
t2 = 4.90;%m
gamma = 0.97;
R_flood4 = zeros(1, n_states);

for i = 1:n_states
        [slr1,surge1] = components(i);
        slrh1 = values_slr(slr1);
        surgeh1 = values_surge(surge1);
        totalh1 = (slrh1 + surgeh1)/100;
        if totalh1 <= seawall_h
            totalh1 = 0;
            area1 = 0;
        elseif totalh1 > seawall_h && totalh1 <= b_green
             f_damage = 1.0;
             totalh1 = totalh1 - seawall_h;
             area1 = 0.5*f_damage*(1/s)*totalh1^2;
        elseif (totalh1> b_green) && (totalh1 <= t_green)
%              l_green_sub = sqrt(((totalh1 - b_green)/s)^2 + (totalh1 - b_green)^2);
%              wa_nbs = (wa_nbs_m)*l_green_sub;
            f_damage1 = 1.00;
            a1 = (0.5*(1/s)*(b_green - seawall_h)^2);
            f_damage2 = 0.80;
            a2 = (0.5*(1/s)*(totalh1-seawall_h)^2 - a1);
            area1 = a1*f_damage1+a2*f_damage2;
        elseif totalh1 > t_green && totalh1<=b2
            f_damage1 = 1.00;
            a1 = (0.5*(1/s)*(b_green-seawall_h)^2);
            f_damage2 = 0.80;
            a2 = (0.5*(1/s)*(t_green-seawall_h)^2 - a1);
            a3 = (0.5*(1/s)*(totalh1-seawall_h)^2 - a1 - a2);
            area1 = a1*f_damage1+a2*f_damage2+a3*f_damage1;
        elseif totalh1>b2 && totalh1<=t2
            f_damage1 = 1.00;
            a1 = (0.5*(1/s)*(b_green-seawall_h)^2);
            f_damage2 = 0.80;
            a2 = (0.5*(1/s)*(t_green-seawall_h)^2 - a1);
            a3 = (0.5*(1/s)*(b2-seawall_h)^2 - a1 - a2);
            a4 = (totalh1 - b2)*(b2-seawall_h)*(1/s);
            area1 = a1*f_damage1+a2*f_damage2+a3*f_damage1+a4*f_damage1;
        elseif totalh1>t2
            f_damage1 = 1.00;
            a1 = (0.5*(1/s)*(b_green-seawall_h)^2);
            f_damage2 = 0.80;
            a2 = (0.5*(1/s)*(t_green-seawall_h)^2 - a1);
            a3 = (0.5*(1/s)*(b2-seawall_h)^2 - a1 - a2);
            a4 = (0.5*(1/s)*(totalh1-seawall_h)^2 - a1 - a2 - a3);
            area1 = a1*f_damage1+a2*f_damage2+a3*f_damage1+a4*f_damage1;
        end

        vol_f(i) = (area1 - 0);%area in m^2
        c_flood = cf*vol_f(i)/vol_z;
        R_flood4(i) = c_flood;
        
end

%risk carbon
R_flood_carbon4 = zeros(40, n_states);
for t = 1:horizon
    c_carbon_flood4 = zeros(1, length(n_states));
for i = 1:n_states
        [~, c_carbon_flood4(i)] = FloodDamageGHGCost(R_flood4(i), discounted_sum_scc(t));      
end
R_flood_carbon4(t,:) = c_carbon_flood4; 
end

%maintenance
R_mc = zeros(40, n_states);
R_m = (cm_green + cm_dike) * ones(1, n_states);
for t = 1:horizon
    Rmc(t) = (cm_carbon_green)*discounted_sum_scc(t) + cm_carbon_dike*discounted_sum_scc(t);
    R_mc(t,:) = ones(1, n_states)*Rmc(t);
end

%carbon uptake
R_up = zeros(40, n_states);
for t = 1:horizon
    Rup(t) = cu_carbon_green*discounted_sum_scc(t);
    R_up(t,:) = ones(1, n_states)*Rup(t);
end

%total
rewards4_dn = cell(1, horizon);
for t = 1:horizon
    rewards4_dn{1,t} = (R_flood4 + R_flood_carbon4(t,:))*gamma + R_up(t, :)+R_m + R_mc(t, :);
end
R_dn{1,4} = rewards4_dn;

% action 2- construct green solution only
R_1 = cell(1, n_systems);

% rewards for system A under action 2 - construct green solution only

slr_states = 77;
surge_states = 72;
n_states = slr_states*surge_states;
b_green = 0.20;%m
t_green = 1.70;%m (dike height = 1.5 m)

R_flood1 = zeros(1, n_states);
for i = 1:n_states
    
        [slr1,surge1] = components(i);
        slrh1 = values_slr(slr1);
        surgeh1 = values_surge(surge1);
        totalh1 = (slrh1 + surgeh1)/100;
        if totalh1 <= seawall_h
            totalh1 = 0;
            area1 = 0;
        elseif totalh1 > seawall_h && totalh1 <= b_green
             f_damage = 1.0;
             totalh1 = totalh1 - seawall_h;
             area1 = 0.5*f_damage*(1/s)*totalh1^2;
        elseif (totalh1> b_green) && (totalh1 <= t_green)
            f_damage1 = 1.00;
            a1 = (0.5*(1/s)*(b_green - seawall_h)^2);
            f_damage2 = 0.80;
            a2 = (0.5*(1/s)*(totalh1-seawall_h)^2 - a1);
            area1 = a1*f_damage1+a2*f_damage2;
        elseif totalh1 > t_green
            f_damage1 = 1.00;
            a1 = (0.5*(1/s)*(b_green-seawall_h)^2);
            f_damage2 = 0.80;
            a2 = (0.5*(1/s)*(t_green-seawall_h)^2 - a1);
            a3 = (0.5*(1/s)*(totalh1-seawall_h)^2 - a1 - a2);
            area1 = a1*f_damage1+a2*f_damage2+a3*f_damage1; 
        end
        
        vol_f(i) = (area1 - 0);%area in m^2
        c_flood = cf*vol_f(i)/vol_z;
        R_flood1(i) = c_flood;

end

%risk carbon
R_flood_carbon1 = zeros(40, n_states);
for t = 1:horizon
    c_carbon_flood1 = zeros(1, length(n_states));
for i = 1:n_states
        [~, c_carbon_flood1(i)] = FloodDamageGHGCost(R_flood1(i), discounted_sum_scc(t));      
end
R_flood_carbon1(t,:) = c_carbon_flood1; 
end

% current construction (carbon gets emitted in the atmosphere)
R_c = cc_green*ones(1, n_states);
R_const = zeros(40, n_states);
for t = 1:horizon
    Rconst(t) = (cc_carbon_green)*discounted_sum_scc(t);
    R_const(t,:) = ones(1, n_states)*Rconst(t);
end

%uptake
R_up = zeros(40, n_states);
for t = 1:horizon
    Rup(t) = cu_carbon_green*discounted_sum_scc(t);
    R_up(t,:) = ones(1, n_states)*Rup(t);
end

%total
rewards1_dike1 = cell(1, 40);
for t = 1:horizon
    rewards1_dike1{1,t} = (R_flood1 + R_flood_carbon1(t,:))*gamma + R_up(t, :) + R_const(t, :) + R_c;
end

R_1{1,1} = rewards1_dike1;

% rewards for system B under action 2 - construct green solution only

slr_states = 77;
surge_states = 72;
n_states = slr_states*surge_states;
b_green = 0.20;%m
t_green = 1.70;%m (dike height = 1.5 m)
R_flood2 = zeros(1, n_states);
for i = 1:n_states
        [slr1,surge1] = components(i);
        slrh1 = values_slr(slr1);
        surgeh1 = values_surge(surge1);
        totalh1 = (slrh1 + surgeh1)/100;
        if totalh1 <= seawall_h
            totalh1 = 0;
            area1 = 0;
        elseif totalh1 > seawall_h && totalh1 <= b_green
             f_damage = 1.0;
             totalh1 = totalh1 - seawall_h;
             area1 = 0.5*f_damage*(1/s)*totalh1^2;
        elseif (totalh1> b_green) && (totalh1 <= t_green)
            f_damage1 = 1.00;
            a1 = (0.5*(1/s)*(b_green - seawall_h)^2);
            f_damage2 = 0.80;
            a2 = (0.5*(1/s)*(totalh1-seawall_h)^2 - a1);
            area1 = a1*f_damage1+a2*f_damage2;
        elseif totalh1 > t_green
            f_damage1 = 1.00;
            a1 = (0.5*(1/s)*(b_green-seawall_h)^2);
            f_damage2 = 0.80;
            a2 = (0.5*(1/s)*(t_green-seawall_h)^2 - a1);
            a3 = (0.5*(1/s)*(totalh1-seawall_h)^2 - a1 - a2);
            area1 = a1*f_damage1+a2*f_damage2+a3*f_damage1; 
        end
        
        vol_f(i) = (area1 - 0);%area in m^2
        c_flood = cf*vol_f(i)/vol_z; 
        R_flood2(i) = c_flood;
end
%risk carbon
R_flood_carbon2 = zeros(40, n_states);
for t = 1:horizon
    c_carbon_flood2 = zeros(1, length(n_states));
for i = 1:n_states
        [~, c_carbon_flood2(i)] = FloodDamageGHGCost(R_flood2(i), discounted_sum_scc(t));      
end
R_flood_carbon2(t,:) = c_carbon_flood2; 
end

% current construction (carbon gets emitted in the atmosphere)
R_c = cc_green*ones(1, n_states);
R_const = zeros(40, n_states);
for t = 1:horizon
    Rconst(t) = (cc_carbon_green)*discounted_sum_scc(t);
    R_const(t,:) = ones(1, n_states)*Rconst(t);
end

%uptake
R_up = zeros(40, n_states);
for t = 1:horizon
    Rup(t) = cu_carbon_green*discounted_sum_scc(t);
    R_up(t,:) = ones(1, n_states)*Rup(t);
end

%maintenance
R_m = cm_green * ones(1, n_states);
R_mc = zeros(40, n_states);
for t = 1:horizon
    Rmc(t) = cm_carbon_green*discounted_sum_scc(t);
    R_mc(t,:) = ones(1, n_states)*Rmc(t);
end

%total
rewards2_dike1 = cell(1, 40);
for t = 1:horizon
    rewards2_dike1{1,t} = (R_flood2 + R_flood_carbon2(t,:))*gamma + R_up(t, :) + R_const(t, :) + R_c + R_m + R_mc(t,:);
end

R_1{1,2} = rewards2_dike1;

% rewards for system C under action 2 

slr_states = 77;
surge_states = 72;
n_states = slr_states*surge_states;
b_green = 0.20;%m
t_green = 1.70;%m
b2 = 1.90;
t2 = 3.40;
R_flood3 = zeros(1, n_states);

for i = 1:n_states
    
        [slr1,surge1] = components(i);
        slrh1 = values_slr(slr1);
        surgeh1 = values_surge(surge1);
        totalh1 = (slrh1 + surgeh1)/100;
        if totalh1 <= seawall_h
            totalh1 = 0;
            area1 = 0;
        elseif totalh1 > seawall_h && totalh1 <= b_green
             f_damage = 1.0;
             totalh1 = totalh1 - seawall_h;
             area1 = 0.5*f_damage*(1/s)*totalh1^2;
        elseif (totalh1> b_green) && (totalh1 <= t_green)
%              l_green_sub = sqrt(((totalh1 - b_green)/s)^2 + (totalh1 - b_green)^2);
%              wa_nbs = (wa_nbs_m)*l_green_sub;
            f_damage1 = 1.00;
            a1 = (0.5*(1/s)*(b_green - seawall_h)^2);
            f_damage2 = 0.80;
            a2 = (0.5*(1/s)*(totalh1-seawall_h)^2 - a1);
            area1 = a1*f_damage1+a2*f_damage2;
        elseif totalh1 > t_green && totalh1<=b2
            f_damage1 = 1.00;
            a1 = (0.5*(1/s)*(b_green-seawall_h)^2);
            f_damage2 = 0.80;
            a2 = (0.5*(1/s)*(t_green-seawall_h)^2 - a1);
            a3 = (0.5*(1/s)*(totalh1-seawall_h)^2 - a1 - a2);
            area1 = a1*f_damage1+a2*f_damage2+a3*f_damage1;
        elseif totalh1>b2 && totalh1<=t2
            f_damage1 = 1.00;
            a1 = (0.5*(1/s)*(b_green-seawall_h)^2);
            f_damage2 = 0.80;
            a2 = (0.5*(1/s)*(t_green-seawall_h)^2 - a1);
            a3 = (0.5*(1/s)*(b2-seawall_h)^2 - a1 - a2);
            a4 = (totalh1 - b2)*(b2-seawall_h)*(1/s);
            area1 = a1*f_damage1+a2*f_damage2+a3*f_damage1+a4*f_damage1;
        elseif totalh1>t2
            f_damage1 = 1.00;
            a1 = (0.5*(1/s)*(b_green-seawall_h)^2);
            f_damage2 = 0.80;
            a2 = (0.5*(1/s)*(t_green-seawall_h)^2 - a1);
            a3 = (0.5*(1/s)*(b2-seawall_h)^2 - a1 - a2);
            a4 = (0.5*(1/s)*(totalh1-seawall_h)^2 - a1 - a2 - a3);
            area1 = a1*f_damage1+a2*f_damage2+a3*f_damage1+a4*f_damage1;
        end
        
        vol_f(i) = (area1 - 0);%area in m^2
        c_flood = cf*vol_f(i)/vol_z;
        R_flood3(i) = c_flood;
end
%risk carbon
R_flood_carbon3 = zeros(40, n_states);
for t = 1:horizon
    c_carbon_flood3 = zeros(1, length(n_states));
for i = 1:n_states
        [~, c_carbon_flood3(i)] = FloodDamageGHGCost(R_flood3(i), discounted_sum_scc(t));      
end
R_flood_carbon3(t,:) = c_carbon_flood3; 
end

% current construction (carbon gets emitted in the atmosphere)
R_c = cc_green*ones(1, n_states);
R_const = zeros(40, n_states);
for t = 1:horizon
    Rconst(t) = (cc_carbon_green)*discounted_sum_scc(t);
    R_const(t,:) = ones(1, n_states)*Rconst(t);
end

%uptake
R_up = zeros(40, n_states);
for t = 1:horizon
    Rup(t) = cu_carbon_green*discounted_sum_scc(t);
    R_up(t,:) = ones(1, n_states)*Rup(t);
end

%maintenance
R_m = cm_dike * ones(1, n_states);
R_mc = zeros(40, n_states);
for t = 1:horizon
    Rmc(t) = cm_carbon_dike*discounted_sum_scc(t);
    R_mc(t,:) = ones(1, n_states)*Rmc(t);
end

%total
rewards3_dike1 = cell(1, 40);
for t = 1:horizon
    rewards3_dike1{1,t} = (R_flood3 + R_flood_carbon3(t,:))*gamma + R_up(t, :) + R_const(t, :) + R_c + R_m + R_mc(t,:);
end
R_1{1,3} = rewards3_dike1;

% rewards for system D under action 2 
slr_states = 77;
surge_states = 72;
n_states = slr_states*surge_states;
b_green = 0.20;%m
t_green = 1.70;%m
b2 = 1.90;
t2 = 3.40;
gamma = 0.97;

R_flood4 =  zeros(1, n_states);
for i = 1:n_states
    
        [slr1,surge1] = components(i);
        slrh1 = values_slr(slr1);
        surgeh1 = values_surge(surge1);
        totalh1 = (slrh1 + surgeh1)/100;
        if totalh1 <= seawall_h
            totalh1 = 0;
            area1 = 0;
        elseif totalh1 > seawall_h && totalh1 <= b_green
             f_damage = 1.0;
             totalh1 = totalh1 - seawall_h;
             area1 = 0.5*f_damage*(1/s)*totalh1^2;
        elseif (totalh1> b_green) && (totalh1 <= t_green)
%              l_green_sub = sqrt(((totalh1 - b_green)/s)^2 + (totalh1 - b_green)^2);
%              wa_nbs = (wa_nbs_m)*l_green_sub;
            f_damage1 = 1.00;
            a1 = (0.5*(1/s)*(b_green - seawall_h)^2);
            f_damage2 = 0.80;
            a2 = (0.5*(1/s)*(totalh1-seawall_h)^2 - a1);
            area1 = a1*f_damage1+a2*f_damage2;
        elseif totalh1 > t_green && totalh1<=b2
            f_damage1 = 1.00;
            a1 = (0.5*(1/s)*(b_green-seawall_h)^2);
            f_damage2 = 0.80;
            a2 = (0.5*(1/s)*(t_green-seawall_h)^2 - a1);
            a3 = (0.5*(1/s)*(totalh1-seawall_h)^2 - a1 - a2);
            area1 = a1*f_damage1+a2*f_damage2+a3*f_damage1;
        elseif totalh1>b2 && totalh1<=t2
            f_damage1 = 1.00;
            a1 = (0.5*(1/s)*(b_green-seawall_h)^2);
            f_damage2 = 0.80;
            a2 = (0.5*(1/s)*(t_green-seawall_h)^2 - a1);
            a3 = (0.5*(1/s)*(b2-seawall_h)^2 - a1 - a2);
            a4 = (totalh1 - b2)*(b2-seawall_h)*(1/s);
            area1 = a1*f_damage1+a2*f_damage2+a3*f_damage1+a4*f_damage1;
        elseif totalh1>t2
            f_damage1 = 1.00;
            a1 = (0.5*(1/s)*(b_green-seawall_h)^2);
            f_damage2 = 0.80;
            a2 = (0.5*(1/s)*(t_green-seawall_h)^2 - a1);
            a3 = (0.5*(1/s)*(b2-seawall_h)^2 - a1 - a2);
            a4 = (0.5*(1/s)*(totalh1-seawall_h)^2 - a1 - a2 - a3);
            area1 = a1*f_damage1+a2*f_damage2+a3*f_damage1+a4*f_damage1;
        end
        
        vol_f(i) = (area1 - 0);%area in m^2
        c_flood = cf*vol_f(i)/vol_z;
        R_flood4(i) = c_flood;

end
%risk carbon
R_flood_carbon4 = zeros(40, n_states);
for t = 1:horizon
    c_carbon_flood4 = zeros(1, length(n_states));
for i = 1:n_states
        [~, c_carbon_flood4(i)] = FloodDamageGHGCost(R_flood4(i), discounted_sum_scc(t));      
end
R_flood_carbon4(t,:) = c_carbon_flood4; 
end
% current construction (carbon gets emitted in the atmosphere)
R_c = cc_green*ones(1, n_states);
R_const = zeros(40, n_states);
for t = 1:horizon
    Rconst(t) = (cc_carbon_green)*discounted_sum_scc(t);
    R_const(t,:) = ones(1, n_states)*Rconst(t);
end

%uptake
R_up = zeros(40, n_states);
for t = 1:horizon
    Rup(t) = cu_carbon_green*discounted_sum_scc(t);
    R_up(t,:) = ones(1, n_states)*Rup(t);
end

%maintenance
R_m = (cm_green + cm_dike) * ones(1, n_states);
R_mc = zeros(40, n_states);
for t = 1:horizon
    Rmc(t) = (cm_carbon_green)*discounted_sum_scc(t) + cm_carbon_dike*discounted_sum_scc(t);
    R_mc(t,:) = ones(1, n_states)*Rmc(t);
end

%total
rewards4_dike1 = cell(1, 40);
for t = 1:horizon
    rewards4_dike1{1,t} = (R_flood4 + R_flood_carbon4(t,:))*gamma + R_up(t, :) + R_const(t, :) + R_c + R_m + R_mc(t,:);
end
R_1{1,4} = rewards4_dike1;


% action 3- construct dike 2 only
R_2 = cell(1,n_systems);

% rewards for system A under action 3
slr_states = 77;
surge_states = 72;
n_states = slr_states*surge_states;
b = 1.90;%cm
t = 3.40;%cm
gamma = 0.97;

R_flood1 = zeros(1, n_states);
for i = 1:n_states
 
        [slr1,surge1] = components(i);
        slrh1 = values_slr(slr1);
        surgeh1 = values_surge(surge1);
        totalh1 = (slrh1 + surgeh1)/100;
        if totalh1 <= seawall_h
            totalh1 = 0;
            area1 = 0;
        elseif (totalh1> b) && (totalh1 <= t)
            area1 = 0.5*(b-seawall_h)*(b-seawall_h)*(1/s) + (totalh1 - b)*(b-seawall_h)*(1/s);
        else
            area1 = 0.5*(1/s)*(totalh1-seawall_h)^2;
        end
        
        
        vol_f(i) = (area1 - 0);%area in m^2
        c_flood = cf*vol_f(i)/vol_z;
        R_flood1(i) = c_flood;

end
%risk carbon
R_flood_carbon1 = zeros(40, n_states);
for t = 1:horizon
    c_carbon_flood1 = zeros(1, length(n_states));
for i = 1:n_states
        [~, c_carbon_flood1(i)] = FloodDamageGHGCost(R_flood1(i), discounted_sum_scc(t));      
end
R_flood_carbon1(t,:) = c_carbon_flood1; 
end

% current construction (carbon gets emitted in the atmosphere)
R_c = cc_dike*ones(1, n_states);
R_const = zeros(40, n_states);
for t = 1:horizon
    Rconst(t) = (cc_carbon_dike)*discounted_sum_scc(t);
    R_const(t,:) = ones(1, n_states)*Rconst(t);
end

%total
rewards1_dike2 = cell(1, 40);
for t = 1:horizon
    rewards1_dike2{1,t} = (R_flood1 + R_flood_carbon1(t,:))*gamma + R_const(t, :) + R_c;
end

R_2{1,1} = rewards1_dike2;

% rewards for system B under action 3
slr_states = 77;
surge_states = 72;
n_states = slr_states*surge_states;
b_green = 0.20;%m
t_green = 1.70;%m
b2 = 1.90;
t2 = 3.40;
gamma = 0.97;

R_flood2 = zeros(1, n_states);
for i = 1:n_states
    
        [slr1,surge1] = components(i);
        slrh1 = values_slr(slr1);
        surgeh1 = values_surge(surge1);
        totalh1 = (slrh1 + surgeh1)/100;
        if totalh1 <= seawall_h
            totalh1 = 0;
            area1 = 0;
        elseif totalh1 > seawall_h && totalh1 <= b_green
             f_damage = 1.0;
             totalh1 = totalh1 - seawall_h;
             area1 = 0.5*f_damage*(1/s)*totalh1^2;
        elseif (totalh1> b_green) && (totalh1 <= t_green)
%              l_green_sub = sqrt(((totalh1 - b_green)/s)^2 + (totalh1 - b_green)^2);
%              wa_nbs = (wa_nbs_m)*l_green_sub;
            f_damage1 = 1.00;
            a1 = (0.5*(1/s)*(b_green - seawall_h)^2);
            f_damage2 = 0.80;
            a2 = (0.5*(1/s)*(totalh1-seawall_h)^2 - a1);
            area1 = a1*f_damage1+a2*f_damage2;
        elseif totalh1 > t_green && totalh1<=b2
            f_damage1 = 1.00;
            a1 = (0.5*(1/s)*(b_green-seawall_h)^2);
            f_damage2 = 0.80;
            a2 = (0.5*(1/s)*(t_green-seawall_h)^2 - a1);
            a3 = (0.5*(1/s)*(totalh1-seawall_h)^2 - a1 - a2);
            area1 = a1*f_damage1+a2*f_damage2+a3*f_damage1;
        elseif totalh1>b2 && totalh1<=t2
            f_damage1 = 1.00;
            a1 = (0.5*(1/s)*(b_green-seawall_h)^2);
            f_damage2 = 0.80;
            a2 = (0.5*(1/s)*(t_green-seawall_h)^2 - a1);
            a3 = (0.5*(1/s)*(b2-seawall_h)^2 - a1 - a2);
            a4 = (totalh1 - b2)*(b2-seawall_h)*(1/s);
            area1 = a1*f_damage1+a2*f_damage2+a3*f_damage1+a4*f_damage1;
        elseif totalh1>t2
            f_damage1 = 1.00;
            a1 = (0.5*(1/s)*(b_green-seawall_h)^2);
            f_damage2 = 0.80;
            a2 = (0.5*(1/s)*(t_green-seawall_h)^2 - a1);
            a3 = (0.5*(1/s)*(b2-seawall_h)^2 - a1 - a2);
            a4 = (0.5*(1/s)*(totalh1-seawall_h)^2 - a1 - a2 - a3);
            area1 = a1*f_damage1+a2*f_damage2+a3*f_damage1+a4*f_damage1;
        end
        vol_f(i) = (area1 - 0);%area in m^2
        c_flood = cf*vol_f(i)/vol_z;
        R_flood2(i) = c_flood;

end
%risk carbon
R_flood_carbon2 = zeros(40, n_states);
for t = 1:horizon
    c_carbon_flood2 = zeros(1, length(n_states));
for i = 1:n_states
        [~, c_carbon_flood2(i)] = FloodDamageGHGCost(R_flood2(i), discounted_sum_scc(t));      
end
R_flood_carbon2(t,:) = c_carbon_flood2; 
end


% current construction (carbon gets emitted in the atmosphere)
R_c = cc_dike*ones(1, n_states);
R_const = zeros(40, n_states);
for t = 1:horizon
    Rconst(t) = (cc_carbon_dike)*discounted_sum_scc(t);
    R_const(t,:) = ones(1, n_states)*Rconst(t);
end

%uptake
R_up = zeros(40, n_states);
for t = 1:horizon
    Rup(t) = cu_carbon_green*discounted_sum_scc(t);
    R_up(t,:) = ones(1, n_states)*Rup(t);
end

%maintenance
R_m = cm_green * ones(1, n_states);
R_mc = zeros(40, n_states);
for t = 1:horizon
    Rmc(t) = cm_carbon_green*discounted_sum_scc(t);
    R_mc(t,:) = ones(1, n_states)*Rmc(t);
end

%total
rewards2_dike2 = cell(1, 40);
for t = 1:horizon
    rewards2_dike2{1,t} = (R_flood2 + R_flood_carbon2(t,:))*gamma + R_up(t, :) + R_const(t, :) + R_c + R_m + R_mc(t,:);
end

R_2{1,2} = rewards2_dike2;

% rewards for system C under action 3

slr_states = 77;
surge_states = 72;
n_states = slr_states*surge_states;
b = 1.90;%cm
t = 3.40;%cm
gamma = 0.97;

R_flood3 = zeros(1, n_states);
for i = 1:n_states
    
        [slr1,surge1] = components(i);
        slrh1 = values_slr(slr1);
        surgeh1 = values_surge(surge1);
        totalh1 = (slrh1 + surgeh1)/100;
        if totalh1 <= seawall_h
            totalh1 = 0;
            area1 = 0;
        elseif (totalh1> b) && (totalh1 <= t)
            area1 = 0.5*(b-seawall_h)*(b-seawall_h)*(1/s) + (totalh1 - b)*(b-seawall_h)*(1/s);
        else
            area1 = 0.5*(1/s)*(totalh1-seawall_h)^2;
        end
        
       
        vol_f(i) = (area1 - 0);%area in m^2
        c_flood = cf*vol_f(i)/vol_z;
        R_flood3(i) = c_flood;

end
%risk carbon
R_flood_carbon3 = zeros(40, n_states);
for t = 1:horizon
    c_carbon_flood3 = zeros(1, length(n_states));
for i = 1:n_states
        [~, c_carbon_flood3(i)] = FloodDamageGHGCost(R_flood3(i), discounted_sum_scc(t));      
end
R_flood_carbon3(t,:) = c_carbon_flood3; 
end

% current construction (carbon gets emitted in the atmosphere)
R_c = cc_dike*ones(1, n_states);
R_const = zeros(40, n_states);
for t = 1:horizon
    Rconst(t) = (cc_carbon_dike)*discounted_sum_scc(t);
    R_const(t,:) = ones(1, n_states)*Rconst(t);
end

%maintenance
R_m = cm_dike * ones(1, n_states);
R_mc = zeros(40, n_states);
for t = 1:horizon
    Rmc(t) = cm_carbon_dike*discounted_sum_scc(t);
    R_mc(t,:) = ones(1, n_states)*Rmc(t);
end

%total
rewards3_dike2 = cell(1, 40);
for t = 1:horizon
    rewards3_dike2{1,t} = (R_flood3 + R_flood_carbon3(t,:))*gamma + R_const(t, :) + R_c + R_m + R_mc(t,:);
end

R_2{1,3} = rewards3_dike2;

% rewards for system D under action 3

slr_states = 77;
surge_states = 72;
n_states = slr_states*surge_states;
b_green = 0.20;%m
t_green = 1.70;%m
b2 = 1.90;
t2 = 3.40;
gamma = 0.97;
R_flood4 = zeros(1,n_states);
for i = 1:n_states
    
        [slr1,surge1] = components(i);
        slrh1 = values_slr(slr1);
        surgeh1 = values_surge(surge1);
        totalh1 = (slrh1 + surgeh1)/100;
        if totalh1 <= seawall_h
            totalh1 = 0;
            area1 = 0;
        elseif totalh1 > seawall_h && totalh1 <= b_green
             f_damage = 1.0;
             totalh1 = totalh1 - seawall_h;
             area1 = 0.5*f_damage*(1/s)*totalh1^2;
        elseif (totalh1> b_green) && (totalh1 <= t_green)
%              l_green_sub = sqrt(((totalh1 - b_green)/s)^2 + (totalh1 - b_green)^2);
%              wa_nbs = (wa_nbs_m)*l_green_sub;
            f_damage1 = 1.00;
            a1 = (0.5*(1/s)*(b_green - seawall_h)^2);
            f_damage2 = 0.80;
            a2 = (0.5*(1/s)*(totalh1-seawall_h)^2 - a1);
            area1 = a1*f_damage1+a2*f_damage2;
        elseif totalh1 > t_green && totalh1<=b2
            f_damage1 = 1.00;
            a1 = (0.5*(1/s)*(b_green-seawall_h)^2);
            f_damage2 = 0.80;
            a2 = (0.5*(1/s)*(t_green-seawall_h)^2 - a1);
            a3 = (0.5*(1/s)*(totalh1-seawall_h)^2 - a1 - a2);
            area1 = a1*f_damage1+a2*f_damage2+a3*f_damage1;
        elseif totalh1>b2 && totalh1<=t2
            f_damage1 = 1.00;
            a1 = (0.5*(1/s)*(b_green-seawall_h)^2);
            f_damage2 = 0.80;
            a2 = (0.5*(1/s)*(t_green-seawall_h)^2 - a1);
            a3 = (0.5*(1/s)*(b2-seawall_h)^2 - a1 - a2);
            a4 = (totalh1 - b2)*(b2-seawall_h)*(1/s);
            area1 = a1*f_damage1+a2*f_damage2+a3*f_damage1+a4*f_damage1;
        elseif totalh1>t2
            f_damage1 = 1.00;
            a1 = (0.5*(1/s)*(b_green-seawall_h)^2);
            f_damage2 = 0.80;
            a2 = (0.5*(1/s)*(t_green-seawall_h)^2 - a1);
            a3 = (0.5*(1/s)*(b2-seawall_h)^2 - a1 - a2);
            a4 = (0.5*(1/s)*(totalh1-seawall_h)^2 - a1 - a2 - a3);
            area1 = a1*f_damage1+a2*f_damage2+a3*f_damage1+a4*f_damage1;
        end
       
        vol_f(i) = (area1 - 0);%area in m^2
        c_flood = cf*vol_f(i)/vol_z;
        R_flood4(i) = c_flood;

end
%risk carbon
R_flood_carbon4 = zeros(40, n_states);
for t = 1:horizon
    c_carbon_flood4 = zeros(1, length(n_states));
for i = 1:n_states
        [~, c_carbon_flood4(i)] = FloodDamageGHGCost(R_flood4(i), discounted_sum_scc(t));      
end
R_flood_carbon4(t,:) = c_carbon_flood4; 
end
% current construction (carbon gets emitted in the atmosphere)
R_c = cc_dike*ones(1, n_states);
R_const = zeros(40, n_states);
for t = 1:horizon
    Rconst(t) = (cc_carbon_dike)*discounted_sum_scc(t);
    R_const(t,:) = ones(1, n_states)*Rconst(t);
end

%uptake
R_up = zeros(40, n_states);
for t = 1:horizon
    Rup(t) = cu_carbon_green*discounted_sum_scc(t);
    R_up(t,:) = ones(1, n_states)*Rup(t);
end

%maintenance
R_m = (cm_green + cm_dike) * ones(1, n_states);
R_mc = zeros(40, n_states);
for t = 1:horizon
    Rmc(t) = (cm_carbon_green + cm_carbon_dike)*discounted_sum_scc(t);
    R_mc(t,:) = ones(1, n_states)*Rmc(t);
end

%total
rewards4_dike2 = cell(1, 40);
for t = 1:horizon
    rewards4_dike2{1,t} = (R_flood4 + R_flood_carbon4(t,:))*gamma + R_up(t, :) + R_const(t, :) + R_c + R_m + R_mc(t,:);
end
R_2{1,4} = rewards4_dike2;

% action 4 - construct both dike and green infra
R_12 = cell(1, n_systems);

% rewards for system A under action 4

slr_states = 77;
surge_states = 72;
n_states = slr_states*surge_states;
b_green = 0.20;%m
t_green = 1.70;%m
b2 = 1.90;
t2 = 3.40;
gamma = 0.97;
R_flood1 = zeros(1, n_states);

for i = 1:n_states
  
        [slr1,surge1] = components(i);
        slrh1 = values_slr(slr1);
        surgeh1 = values_surge(surge1);
        totalh1 = (slrh1 + surgeh1)/100;
        if totalh1 <= seawall_h
            totalh1 = 0;
            area1 = 0;
        elseif totalh1 > seawall_h && totalh1 <= b_green
             f_damage = 1.0;
             totalh1 = totalh1 - seawall_h;
             area1 = 0.5*f_damage*(1/s)*totalh1^2;
        elseif (totalh1> b_green) && (totalh1 <= t_green)
%              l_green_sub = sqrt(((totalh1 - b_green)/s)^2 + (totalh1 - b_green)^2);
%              wa_nbs = (wa_nbs_m)*l_green_sub;
            f_damage1 = 1.00;
            a1 = (0.5*(1/s)*(b_green - seawall_h)^2);
            f_damage2 = 0.80;
            a2 = (0.5*(1/s)*(totalh1-seawall_h)^2 - a1);
            area1 = a1*f_damage1+a2*f_damage2;
        elseif totalh1 > t_green && totalh1<=b2
            f_damage1 = 1.00;
            a1 = (0.5*(1/s)*(b_green-seawall_h)^2);
            f_damage2 = 0.80;
            a2 = (0.5*(1/s)*(t_green-seawall_h)^2 - a1);
            a3 = (0.5*(1/s)*(totalh1-seawall_h)^2 - a1 - a2);
            area1 = a1*f_damage1+a2*f_damage2+a3*f_damage1;
        elseif totalh1>b2 && totalh1<=t2
            f_damage1 = 1.00;
            a1 = (0.5*(1/s)*(b_green-seawall_h)^2);
            f_damage2 = 0.80;
            a2 = (0.5*(1/s)*(t_green-seawall_h)^2 - a1);
            a3 = (0.5*(1/s)*(b2-seawall_h)^2 - a1 - a2);
            a4 = (totalh1 - b2)*(b2-seawall_h)*(1/s);
            area1 = a1*f_damage1+a2*f_damage2+a3*f_damage1+a4*f_damage1;
        elseif totalh1>t2
            f_damage1 = 1.00;
            a1 = (0.5*(1/s)*(b_green-seawall_h)^2);
            f_damage2 = 0.80;
            a2 = (0.5*(1/s)*(t_green-seawall_h)^2 - a1);
            a3 = (0.5*(1/s)*(b2-seawall_h)^2 - a1 - a2);
            a4 = (0.5*(1/s)*(totalh1-seawall_h)^2 - a1 - a2 - a3);
            area1 = a1*f_damage1+a2*f_damage2+a3*f_damage1+a4*f_damage1;
        end
        
        vol_f(i) = (area1 - 0);%area in m^2
        c_flood = cf*vol_f(i)/vol_z;
        R_flood1(i) = c_flood;
end
%risk carbon
R_flood_carbon1 = zeros(40, n_states);
for t = 1:horizon
    c_carbon_flood1 = zeros(1, length(n_states));
for i = 1:n_states
        [~, c_carbon_flood1(i)] = FloodDamageGHGCost(R_flood1(i), discounted_sum_scc(t));      
end
R_flood_carbon1(t,:) = c_carbon_flood1; 
end

% current construction (carbon gets emitted in the atmosphere)
R_c = (cc_dike + cc_green)*ones(1, n_states);
R_const = zeros(40, n_states);
for t = 1:horizon
    Rconst(t) = (cc_carbon_dike + cc_carbon_green)*discounted_sum_scc(t);
    R_const(t,:) = ones(1, n_states)*Rconst(t);
end

%uptake
R_up = zeros(40, n_states);
for t = 1:horizon
    Rup(t) = cu_carbon_green*discounted_sum_scc(t);
    R_up(t,:) = ones(1, n_states)*Rup(t);
end

%total
rewards1_dikes12 = cell(1, 40);
for t = 1:horizon
    rewards1_dikes12{1,t} = (R_flood1 + R_flood_carbon1(t,:))*gamma + R_up(t, :) + R_const(t, :) + R_c;
end

R_12{1,1} = rewards1_dikes12;
% rewards for system B under action 4

slr_states = 77;
surge_states = 72;
n_states = slr_states*surge_states;
b_green = 0.20;%m
t_green = 1.70;%m
b2 = 1.90;
t2 = 3.40;
gamma = 0.97;

R_flood2 = zeros(1, n_states);
for i = 1:n_states
   
        [slr1,surge1] = components(i);
        slrh1 = values_slr(slr1);
        surgeh1 = values_surge(surge1);
        totalh1 = (slrh1 + surgeh1)/100;
        if totalh1 <= seawall_h
            totalh1 = 0;
            area1 = 0;
        elseif totalh1 > seawall_h && totalh1 <= b_green
             f_damage = 1.0;
             totalh1 = totalh1 - seawall_h;
             area1 = 0.5*f_damage*(1/s)*totalh1^2;
        elseif (totalh1> b_green) && (totalh1 <= t_green)
%              l_green_sub = sqrt(((totalh1 - b_green)/s)^2 + (totalh1 - b_green)^2);
%              wa_nbs = (wa_nbs_m)*l_green_sub;
            f_damage1 = 1.00;
            a1 = (0.5*(1/s)*(b_green - seawall_h)^2);
            f_damage2 = 0.80;
            a2 = (0.5*(1/s)*(totalh1-seawall_h)^2 - a1);
            area1 = a1*f_damage1+a2*f_damage2;
        elseif totalh1 > t_green && totalh1<=b2
            f_damage1 = 1.00;
            a1 = (0.5*(1/s)*(b_green-seawall_h)^2);
            f_damage2 = 0.80;
            a2 = (0.5*(1/s)*(t_green-seawall_h)^2 - a1);
            a3 = (0.5*(1/s)*(totalh1-seawall_h)^2 - a1 - a2);
            area1 = a1*f_damage1+a2*f_damage2+a3*f_damage1;
        elseif totalh1>b2 && totalh1<=t2
            f_damage1 = 1.00;
            a1 = (0.5*(1/s)*(b_green-seawall_h)^2);
            f_damage2 = 0.80;
            a2 = (0.5*(1/s)*(t_green-seawall_h)^2 - a1);
            a3 = (0.5*(1/s)*(b2-seawall_h)^2 - a1 - a2);
            a4 = (totalh1 - b2)*(b2-seawall_h)*(1/s);
            area1 = a1*f_damage1+a2*f_damage2+a3*f_damage1+a4*f_damage1;
        elseif totalh1>t2
            f_damage1 = 1.00;
            a1 = (0.5*(1/s)*(b_green-seawall_h)^2);
            f_damage2 = 0.80;
            a2 = (0.5*(1/s)*(t_green-seawall_h)^2 - a1);
            a3 = (0.5*(1/s)*(b2-seawall_h)^2 - a1 - a2);
            a4 = (0.5*(1/s)*(totalh1-seawall_h)^2 - a1 - a2 - a3);
            area1 = a1*f_damage1+a2*f_damage2+a3*f_damage1+a4*f_damage1;
        end
        vol_f(i) = (area1 - 0);%area in m^2
        c_flood = cf*vol_f(i)/vol_z;
        R_flood2(i) = c_flood;

end
%risk carbon
R_flood_carbon2 = zeros(40, n_states);
for t = 1:horizon
    c_carbon_flood2 = zeros(1, length(n_states));
for i = 1:n_states
        [~, c_carbon_flood2(i)] = FloodDamageGHGCost(R_flood2(i), discounted_sum_scc(t));      
end
R_flood_carbon2(t,:) = c_carbon_flood2; 
end
% current construction (carbon gets emitted in the atmosphere)
R_c = (cc_dike+cc_green)*ones(1, n_states);
R_const = zeros(40, n_states);
for t = 1:horizon
    Rconst(t) = (cc_carbon_dike+cc_carbon_green)*discounted_sum_scc(t);
    R_const(t,:) = ones(1, n_states)*Rconst(t);
end

%uptake
R_up = zeros(40, n_states);
for t = 1:horizon
    Rup(t) = cu_carbon_green*discounted_sum_scc(t);
    R_up(t,:) = ones(1, n_states)*Rup(t);
end

%maintenance
R_m = cm_green * ones(1, n_states);
R_mc = zeros(40, n_states);
for t = 1:horizon
    Rmc(t) = cm_carbon_green*discounted_sum_scc(t);
    R_mc(t,:) = ones(1, n_states)*Rmc(t);
end

%total
rewards2_dikes12 = cell(1, 40);
for t = 1:horizon
    rewards2_dikes12{1,t} = (R_flood2 + R_flood_carbon2(t,:))*gamma + R_up(t, :) + R_const(t, :) + R_c + R_m + R_mc(t,:);
end

R_12{1,2} = rewards2_dikes12;

% rewards for system C under action 4

slr_states = 77;
surge_states = 72;
n_states = slr_states*surge_states;
b_green = 0.20;%m
t_green = 1.70;%m
b2 = 1.90;
t2 = 3.40;
gamma = 0.97;

R_flood3 = zeros(1,n_states);
for i = 1:n_states

        [slr1,surge1] = components(i);
        slrh1 = values_slr(slr1);
        surgeh1 = values_surge(surge1);
        totalh1 = (slrh1 + surgeh1)/100;
        if totalh1 <= seawall_h
            totalh1 = 0;
            area1 = 0;
        elseif totalh1 > seawall_h && totalh1 <= b_green
             f_damage = 1.0;
             totalh1 = totalh1 - seawall_h;
             area1 = 0.5*f_damage*(1/s)*totalh1^2;
        elseif (totalh1> b_green) && (totalh1 <= t_green)
%              l_green_sub = sqrt(((totalh1 - b_green)/s)^2 + (totalh1 - b_green)^2);
%              wa_nbs = (wa_nbs_m)*l_green_sub;
            f_damage1 = 1.00;
            a1 = (0.5*(1/s)*(b_green - seawall_h)^2);
            f_damage2 = 0.80;
            a2 = (0.5*(1/s)*(totalh1-seawall_h)^2 - a1);
            area1 = a1*f_damage1+a2*f_damage2;
        elseif totalh1 > t_green && totalh1<=b2
            f_damage1 = 1.00;
            a1 = (0.5*(1/s)*(b_green-seawall_h)^2);
            f_damage2 = 0.80;
            a2 = (0.5*(1/s)*(t_green-seawall_h)^2 - a1);
            a3 = (0.5*(1/s)*(totalh1-seawall_h)^2 - a1 - a2);
            area1 = a1*f_damage1+a2*f_damage2+a3*f_damage1;
        elseif totalh1>b2 && totalh1<=t2
            f_damage1 = 1.00;
            a1 = (0.5*(1/s)*(b_green-seawall_h)^2);
            f_damage2 = 0.80;
            a2 = (0.5*(1/s)*(t_green-seawall_h)^2 - a1);
            a3 = (0.5*(1/s)*(b2-seawall_h)^2 - a1 - a2);
            a4 = (totalh1 - b2)*(b2-seawall_h)*(1/s);
            area1 = a1*f_damage1+a2*f_damage2+a3*f_damage1+a4*f_damage1;
        elseif totalh1>t2
            f_damage1 = 1.00;
            a1 = (0.5*(1/s)*(b_green-seawall_h)^2);
            f_damage2 = 0.80;
            a2 = (0.5*(1/s)*(t_green-seawall_h)^2 - a1);
            a3 = (0.5*(1/s)*(b2-seawall_h)^2 - a1 - a2);
            a4 = (0.5*(1/s)*(totalh1-seawall_h)^2 - a1 - a2 - a3);
            area1 = a1*f_damage1+a2*f_damage2+a3*f_damage1+a4*f_damage1;
        end
        
        vol_f(i) = (area1 - 0);%area in m^2
        c_flood = cf*vol_f(i)/vol_z;
        R_flood3(i) = c_flood;

end
%risk carbon
R_flood_carbon3 = zeros(40, n_states);
for t = 1:horizon
    c_carbon_flood3 = zeros(1, length(n_states));
for i = 1:n_states
        [~, c_carbon_flood3(i)] = FloodDamageGHGCost(R_flood3(i), discounted_sum_scc(t));      
end
R_flood_carbon3(t,:) = c_carbon_flood3; 
end
% current construction (carbon gets emitted in the atmosphere)
R_c = (cc_dike + cc_green)*ones(1, n_states);
R_const = zeros(40, n_states);
for t = 1:horizon
    Rconst(t) = (cc_carbon_dike+cc_carbon_green)*discounted_sum_scc(t);
    R_const(t,:) = ones(1, n_states)*Rconst(t);
end

%maintenance
R_m = cm_dike * ones(1, n_states);
R_mc = zeros(40, n_states);
for t = 1:horizon
    Rmc(t) = cm_carbon_dike*discounted_sum_scc(t);
    R_mc(t,:) = ones(1, n_states)*Rmc(t);
end
%uptake
R_up = zeros(40, n_states);
for t = 1:horizon
    Rup(t) = cu_carbon_green*discounted_sum_scc(t);
    R_up(t,:) = ones(1, n_states)*Rup(t);
end
%total
rewards3_dikes12 = cell(1, 40);
for t = 1:horizon
    rewards3_dikes12{1,t} = (R_flood3 + R_flood_carbon3(t,:))*gamma + R_const(t, :) + R_c + R_m + R_mc(t,:) + R_up(t,:);
end

R_12{1,3} = rewards3_dikes12;


% rewards for system D under action 4

slr_states = 77;
surge_states = 72;
n_states = slr_states*surge_states;
b_green = 0.20;%m
t_green = 1.70;%m
b2 = 1.90;
t2 = 3.40;
gamma = 0.97;

R_flood4 = zeros(1, n_states);
for i = 1:n_states
    
        [slr1,surge1] = components(i);
        slrh1 = values_slr(slr1);
        surgeh1 = values_surge(surge1);
        totalh1 = (slrh1 + surgeh1)/100;
        if totalh1 <= seawall_h
            totalh1 = 0;
            area1 = 0;
        elseif totalh1 > seawall_h && totalh1 <= b_green
             f_damage = 1.0;
             totalh1 = totalh1 - seawall_h;
             area1 = 0.5*f_damage*(1/s)*totalh1^2;
        elseif (totalh1> b_green) && (totalh1 <= t_green)
%              l_green_sub = sqrt(((totalh1 - b_green)/s)^2 + (totalh1 - b_green)^2);
%              wa_nbs = (wa_nbs_m)*l_green_sub;
            f_damage1 = 1.00;
            a1 = (0.5*(1/s)*(b_green - seawall_h)^2);
            f_damage2 = 0.80;
            a2 = (0.5*(1/s)*(totalh1-seawall_h)^2 - a1);
            area1 = a1*f_damage1+a2*f_damage2;
        elseif totalh1 > t_green && totalh1<=b2
            f_damage1 = 1.00;
            a1 = (0.5*(1/s)*(b_green-seawall_h)^2);
            f_damage2 = 0.80;
            a2 = (0.5*(1/s)*(t_green-seawall_h)^2 - a1);
            a3 = (0.5*(1/s)*(totalh1-seawall_h)^2 - a1 - a2);
            area1 = a1*f_damage1+a2*f_damage2+a3*f_damage1;
        elseif totalh1>b2 && totalh1<=t2
            f_damage1 = 1.00;
            a1 = (0.5*(1/s)*(b_green-seawall_h)^2);
            f_damage2 = 0.80;
            a2 = (0.5*(1/s)*(t_green-seawall_h)^2 - a1);
            a3 = (0.5*(1/s)*(b2-seawall_h)^2 - a1 - a2);
            a4 = (totalh1 - b2)*(b2-seawall_h)*(1/s);
            area1 = a1*f_damage1+a2*f_damage2+a3*f_damage1+a4*f_damage1;
        elseif totalh1>t2
            f_damage1 = 1.00;
            a1 = (0.5*(1/s)*(b_green-seawall_h)^2);
            f_damage2 = 0.80;
            a2 = (0.5*(1/s)*(t_green-seawall_h)^2 - a1);
            a3 = (0.5*(1/s)*(b2-seawall_h)^2 - a1 - a2);
            a4 = (0.5*(1/s)*(totalh1-seawall_h)^2 - a1 - a2 - a3);
            area1 = a1*f_damage1+a2*f_damage2+a3*f_damage1+a4*f_damage1;
        end
         
        vol_f(i) = (area1 - 0);%area in m^2
        c_flood = cf*vol_f(i)/vol_z;
        R_flood4(i) = c_flood;
end

%risk carbon
R_flood_carbon4 = zeros(40, n_states);
for t = 1:horizon
    c_carbon_flood4 = zeros(1, length(n_states));
for i = 1:n_states
        [~, c_carbon_flood4(i)] = FloodDamageGHGCost(R_flood4(i), discounted_sum_scc(t));      
end
R_flood_carbon4(t,:) = c_carbon_flood4; 
end
% current construction (carbon gets emitted in the atmosphere)
R_c = (cc_dike+cc_green)*ones(1, n_states);
R_const = zeros(40, n_states);
for t = 1:horizon
    Rconst(t) = (cc_carbon_dike+cc_carbon_green)*discounted_sum_scc(t);
    R_const(t,:) = ones(1, n_states)*Rconst(t);
end

%uptake
R_up = zeros(40, n_states);
for t = 1:horizon
    Rup(t) = cu_carbon_green*scc(t);
    R_up(t,:) = ones(1, n_states)*Rup(t);
end

%maintenance
R_m = (cm_green + cm_dike) * ones(1, n_states);
R_mc = zeros(40, n_states);
for t = 1:horizon
    Rmc(t) = (cm_carbon_green + cm_carbon_dike)*discounted_sum_scc(t);
    R_mc(t,:) = ones(1, n_states)*Rmc(t);
end

%total
rewards4_dikes12 = cell(1, 40);
for t = 1:horizon
    rewards4_dikes12{1,t} = (R_flood4 + R_flood_carbon4(t,:))*gamma + R_up(t, :) + R_const(t, :) + R_c + R_m + R_mc(t,:);
end
R_12{1,4} = rewards4_dikes12;


