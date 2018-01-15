%{
nohup matlab -nodesktop -nodisplay < main002_instSeg_v1_ftAbsEucMM_epoch83.m 1>main002_instSeg_v1_ftAbsEucMM_epoch83.log &
%}
clear
% close all
clc;

addpath './fun4MeanShift';
addpath('./local_functions_demo1');
addpath '../libs/exportFig/';
addpath '../libs/layerExt/';
addpath '../libs/myFunctions/';

path_to_matconvnet = '../libs/matconvnet-1.0-beta23_modifiedDagnn';

run(fullfile(path_to_matconvnet, 'matlab', 'vl_setupnn'));
addpath(genpath(fullfile('dependencies', 'matconvnet','examples')));
%% read matconvnet model
load('imdb_toydata_v3_from_mnist.mat');
imdb.path = './toydata_v3';
imdb.path_to_dataset = './toydata_v3';
% set GPU
gpuId = 1; 
gpuDevice(gpuId);
flagSaveFig = true; % {true false} whether to store the result

saveFolder = 'main007_instSeg_v1_absEucMM';
modelName = 'softmax_net-epoch-83.mat';



netbasemodel = load( fullfile('./exp', saveFolder, modelName) );
netbasemodel = netbasemodel.net;

netbasemodel.layers(136).block = rmfield(netbasemodel.layers(136).block, 'ignoreAverage');
netbasemodel.layers(135).block = rmfield(netbasemodel.layers(135).block, 'ignoreAverage');

netbasemodel = dagnn.DagNN.loadobj(netbasemodel);
%% 1st mean-shift grouping loop
keepLayerName = sprintf('obj_instSeg_reg');
netbasemodel.layers(netbasemodel.getLayerIndex(keepLayerName)).block.lastLayerName = 'res7_cosSim';
keepLayerName = sprintf('obj_instSeg_MM');
netbasemodel.layers(netbasemodel.getLayerIndex(keepLayerName)).block.lastLayerName = 'res7_cosSim';

weight_for_losses = {'obj_instSeg_reg', 1, 'obj_instSeg_MM', 1};
sName_l2norm = 'res7_l2norm';
gt_name =  sprintf('gt_ins');
GaussianBandwidth = 0.1;
randSampleRatio = 0.2;
for loopIdx = 1:5
    [netbasemodel, sName, sName_l2norm] = addOneLoop_forMeanShiftGrouping(...
        netbasemodel, sName_l2norm, loopIdx,...
        GaussianBandwidth, randSampleRatio);
    
    % add regression loss
    obj_name = sprintf('loop%d_instSeg_reg', loopIdx);
    netbasemodel.addLayer(obj_name, ...
        InstanceSegRegLoss_randSample('loss', 'cosinesimilarityabsregloss', 'lastLayerName', sName), ... softmaxlog logistic
        {sName, gt_name}, obj_name);
    weight_for_losses{end+1} = obj_name;
    weight_for_losses{end+1} = 1;
    
    % add max-margin loss
    obj_name = sprintf('loop%d_instSeg_MM', loopIdx);    
    netbasemodel.addLayer(obj_name, ...
        InstanceSegMMLoss_randSample('loss', 'cosinesimilaritymmloss', 'marginAlpha_', 0.07, 'adaptiveMM', false, 'lastLayerName', sName), ...
        {sName, gt_name}, obj_name)
    weight_for_losses{end+1} = obj_name;
    weight_for_losses{end+1} = 1; 
end
%% show learning rates for all layers
for ii = 1:numel(netbasemodel.layers)    
    curLayerName = netbasemodel.layers(ii).name;
    if strfind(curLayerName, 'bn')
        fprintf('%03d, %s\n', ii, curLayerName);
        netbasemodel.params(netbasemodel.layers(ii).paramIndexes(3)).learningRate = 0.1;
    end
end
netbasemodel.params(netbasemodel.getParamIndex('res6_conv_f')).learningRate = 1;
netbasemodel.params(netbasemodel.getParamIndex('res6_conv_b')).learningRate = 1;

for i = 1:numel(netbasemodel.params)
    fprintf('%d\t%25s, \t%.2f',i, netbasemodel.params(i).name, netbasemodel.params(i).learningRate);
    fprintf('\tsize: %dx%dx%dx%d\n', size(netbasemodel.params(i).value,1), size(netbasemodel.params(i).value,2), size(netbasemodel.params(i).value,3), size(netbasemodel.params(i).value,4));
end
%% configure training environment
batchSize = 1;
totalEpoch = 100;
learningRate = 1:totalEpoch;
learningRate = (1.0e-4) * (1-learningRate/totalEpoch).^0.9;

weightDecay=0.0005; % weightDecay: usually use the default value

opts.batchSize = batchSize;
opts.learningRate = learningRate;
opts.weightDecay = weightDecay;
opts.momentum = 0.9 ;

opts.expDir = fullfile('./exp', 'main002_instSeg_v1_ftAbsEucMM_epoch83');
if ~isdir(opts.expDir)
    mkdir(opts.expDir);
end

opts.withSemanticSeg = false ;
opts.withInstanceSeg = true ;
opts.withWeights = false ;

opts.numSubBatches = 1 ;
opts.continue = true ;
opts.gpus = gpuId ;
%gpuDevice(opts.train.gpus); % don't want clear the memory
opts.prefetch = false ;
opts.sync = false ; % for speed
opts.cudnn = true ; % for speed
opts.numEpochs = numel(opts.learningRate) ;
opts.learningRate = learningRate;

for i = 1:2
    curSetName = imdb.sets.name{i};
    idxList = find(imdb.set==i);
    curList = imdb.imgList(idxList);
    opts.(curSetName) = curList;    
end

opts.checkpointFn = [];
mopts.classifyType = 'softmax';

rng(777);
bopts = netbasemodel.meta.normalization;
bopts.numThreads = 12;
bopts.imdb = imdb;
%% train
fn = getBatchWrapper4toyDigitV2(bopts);

opts.backPropDepth = inf; % could limit the backprop
prefixStr = [mopts.classifyType, '_'];
opts.backPropAboveLayerName = 'conv1_conv'; 

trainfn = @cnnTrain;
[netbasemodel, info] = trainfn(netbasemodel, prefixStr, imdb, fn, 'derOutputs', ...
    weight_for_losses, ...
    opts);

%% leaving blank
%{
%}



