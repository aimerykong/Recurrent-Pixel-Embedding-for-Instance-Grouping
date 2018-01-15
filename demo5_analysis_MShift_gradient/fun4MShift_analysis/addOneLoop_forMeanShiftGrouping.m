function [net, sName] = addOneLoop_forMeanShiftGrouping(net, sName, loopIdx, GaussianBandwidth)

if ~exist('GaussianBandwidth', 'var')
    GaussianBandwidth = 0.1;
end
%%
pre_input_layer = sName;

lName = sprintf('loop%d_meanshift_S_is_XX', loopIdx);
net.addLayer(lName, ...
    meanshift_S_is_XX(), ... 
    {sName}, lName);
sName = lName;

lName = sprintf('loop%d_meanshift_G_is_Gaussian', loopIdx);
net.addLayer(lName, ...
    meanshift_G_is_Gaussian('delta', GaussianBandwidth), ...
    {sName}, lName);
G_layer = lName;
sName = lName;


lName = sprintf('loop%d_meanshift_d_is_sumG', loopIdx);
net.addLayer(lName, ...
    meanshift_d_is_sumG(), ...
    {sName}, lName);
sName = lName;


lName = sprintf('loop%d_meanshift_q_is_inv_d', loopIdx);
net.addLayer(lName, ...
    meanshift_q_is_inv_d(), ...
    {sName}, lName);
sName = lName;


lName = sprintf('loop%d_meanshift_P_is_G_diag_q', loopIdx);
net.addLayer(lName, ...
    meanshift_P_is_G_diag_q(), ...
    {G_layer, sName}, lName);
sName = lName;


lName = sprintf('loop%d_meanshift_Y_is_XP', loopIdx);
net.addLayer(lName, ...
    meanshift_Y_is_XP(), ...
    {pre_input_layer, sName}, lName);
sName = lName;


