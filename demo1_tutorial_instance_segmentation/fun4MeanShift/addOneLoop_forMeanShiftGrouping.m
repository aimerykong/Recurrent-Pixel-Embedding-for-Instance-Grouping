function [net, sName, sName_l2norm] = addOneLoop_forMeanShiftGrouping(net, sName, loopIdx, GaussianBandwidth, randSampleRatio)

if ~exist('GaussianBandwidth', 'var')
    GaussianBandwidth = 0.1;
end

if ~exist('randSampleRatio', 'var')
    randSampleRatio = 0.2;    
end

%%
pre_l2_norm_layer = sName;

lName = sprintf('loop%d_meanshift_S_is_XX', loopIdx);
net.addLayer(lName, ...
    meanshift_S_is_XX(), ... softmaxlog logistic
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
    {pre_l2_norm_layer, sName}, lName);
sName = lName;


lName = sprintf('loop%d_meanshift_Y_l2norm', loopIdx);
net.addLayer(lName, L2normalization(), sName, lName) ;
sName = lName;
sName_l2norm = lName;


lName = sprintf('loop%d_meanshift_cosSim', loopIdx);
gt_name =  sprintf('gt_ins');
net.addLayer(lName, cosineSimilarity_randSample('randSampleRatio', randSampleRatio), {sName, gt_name}, lName) ;
sName = lName;


% % add regression loss
% obj_name = sprintf('loop%d_instSeg_reg', loopIdx);
% net.addLayer(obj_name, ...
%     InstanceSegRegLoss_randSample('loss', 'cosinesimilarityabsregloss', 'lastLayerName', sName), ... softmaxlog logistic
%     {sName, gt_name}, obj_name);
% 
% 
% % add max-margin loss
% obj_name = sprintf('loop%d_instSeg_MM', loopIdx);
% input_name = sName;
% net.addLayer(obj_name, ...
%     InstanceSegMMLoss_randSample('loss', 'cosinesimilaritymmloss', 'marginAlpha_', 0.1, 'adaptiveMM', false, 'lastLayerName', sName), ...
%     {input_name, gt_name}, obj_name)