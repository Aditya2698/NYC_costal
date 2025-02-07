% wave attenuation model for oyester reefs : nature based solution
% at edge of the wedge

% wave attenuation model: 40% for total water depth of 0.5–1.0 m; 10% for water depth of 1.0–1.5 m; assume 5% for >1.5 m depth

function wa = wa_oyester(total_h)
if total_h <= 0.5
    wa = 0.50;
elseif total_h > 0.5 && total_h <= 1.00
    wa = 0.40;
elseif total_h > 1.00 && total_h <= 1.50
    wa = 0.10;
elseif total_h > 1.50
    wa = 0.05;
end
    