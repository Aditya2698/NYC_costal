% wave attenuation model for salt marshes : nature based solution
% at edge of the wedge

% wave attenuation model: 60% for total water depth less than 0.1m; y = 6.3398*x^(-0.974) for >=0.1 m depth

function wa = wa_saltmarsh(total_h)
if total_h <= 0.1
    wa = 0.60; % fraction
elseif total_h >= 0.1 
    wa = (6.3398*(total_h)^(-0.974))/100; % fraction
end
    