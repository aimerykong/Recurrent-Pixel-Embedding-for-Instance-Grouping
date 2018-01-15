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

saveFolder = 'semantic_main003_seg_v4_largeLR_l2norm_toyDigitV3';
modelName = 'softmax_net-epoch-63.mat';

netbasemodel = load( fullfile('./exp', saveFolder, modelName) );
netbasemodel = netbasemodel.net;

netbasemodel.layers(137).block = rmfield(netbasemodel.layers(137).block, 'ignoreAverage');
netbasemodel.layers(138).block = rmfield(netbasemodel.layers(138).block, 'ignoreAverage');

netbasemodel = dagnn.DagNN.loadobj(netbasemodel);
%% modify layers, e.g. removing
netbasemodel.removeLayer('obj_segAcc');
netbasemodel.removeLayer('obj_seg');
netbasemodel.removeLayer('res8_conv');
netbasemodel.removeLayer('res7_l2norm');
netbasemodel.removeLayer('res6_relu');
netbasemodel.removeLayer('res6_bn');
netbasemodel.removeLayer('res6_conv');

netbasemodel.meta.normalization.averageImage = imdb.meta.meanvalue; 
netbasemodel.meta.normalization.imageSize = [imdb.meta.height, imdb.meta.width, 1, 1];
%% add a convolution layer
sName = 'conv5_3';
baseName = 'res6';
kernelSZ = [3 3 512 3];
stride = 1;
pad = 1;
hasBias = true;
dilate = 1;
lName = [baseName '_conv'];
block = dagnn.Conv('size', kernelSZ, 'hasBias', hasBias, 'stride', stride, 'pad', pad, 'dilate', dilate);
netbasemodel.addLayer(lName, block, sName, lName, {[lName '_f'], [lName '_b']});
filter = randn(kernelSZ, 'single')*sqrt(2/kernelSZ(end));
netbasemodel.params(netbasemodel.getParamIndex([lName '_f'])).value = filter;
netbasemodel.params(netbasemodel.getParamIndex([lName '_f'])).weightDecay = 1;
netbasemodel.params(netbasemodel.getParamIndex([lName '_f'])).learningRate = 1;
netbasemodel.params(netbasemodel.getParamIndex([lName '_b'])).value = zeros([1 kernelSZ(4)], 'single');
netbasemodel.params(netbasemodel.getParamIndex([lName '_b'])).learningRate = 2;
sName = lName;
%% add L2 normalization layer
lName = 'res7_l2norm';
netbasemodel.addLayer(lName, L2normalization(), sName, lName) ;
x_l2_norm_layer = lName;
sName = lName;
%% add cosineSimilarity layer
lName = 'res7_cosSim';
gt_name =  sprintf('gt_ins');
netbasemodel.addLayer(lName, cosineSimilarity_randSample('randSampleRatio', 0.2), {sName, gt_name}, lName) ;
sName = lName;
%% add regression loss
% add regression loss
obj_name = sprintf('obj_instSeg_reg');
netbasemodel.addLayer(obj_name, ...
    InstanceSegRegLoss_randSample('loss', 'cosinesimilarityabsregloss', 'lastLayerName', sName), ... softmaxlog logistic
    {sName, gt_name}, obj_name);

% add max-margin loss
obj_name = sprintf('obj_instSeg_MM');
input_name = sName;
netbasemodel.addLayer(obj_name, ...
    InstanceSegMMLoss_randSample('loss', 'cosinesimilaritymmloss', 'marginAlpha_', 0.1, 'adaptiveMM', false, 'lastLayerName', sName), ...
    {input_name, gt_name}, obj_name)
%% 1st mean-shift grouping loop
weight_for_losses = {'obj_instSeg_reg', 1, 'obj_instSeg_MM', 1};
sName_l2norm = 'res7_l2norm';
for loopIdx = 1:5
    [netbasemodel, sName, sName_l2norm] = addOneLoop_forMeanShiftGrouping(netbasemodel, sName_l2norm, loopIdx);
    
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
        InstanceSegMMLoss_randSample('loss', 'cosinesimilaritymmloss', 'marginAlpha_', 0.1, 'adaptiveMM', false, 'lastLayerName', sName), ...
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

for i = 125:numel(netbasemodel.params)
    if ~isempty(strfind(netbasemodel.params(i).name, '_f')) && isempty(strfind(netbasemodel.params(i).name, '_bn'))
        tmp = netbasemodel.params(i).value;
        tmp = single(randn(size(tmp), 'single')*sqrt(2/size(tmp,4)));
        netbasemodel.params(i).value = tmp;
        fprintf('%s \n', netbasemodel.params(i).name);
    elseif ~isempty(strfind(netbasemodel.params(i).name, '_b')) && isempty(strfind(netbasemodel.params(i).name, '_bn'))
        tmp = netbasemodel.params(i).value;        
        netbasemodel.params(i).value = zeros(1, size(tmp,2), 'single');
        fprintf('%s \n', netbasemodel.params(i).name);
    elseif ~isempty(strfind(netbasemodel.params(i).name, '_bn')) && ~isempty(strfind(netbasemodel.params(i).name, '_w'))
        tmp = netbasemodel.params(i).value;
        tmp = ones(size(tmp), 'single');
        netbasemodel.params(i).value = tmp;
        fprintf('%s \n', netbasemodel.params(i).name);
    elseif ~isempty(strfind(netbasemodel.params(i).name, '_bn')) && ~isempty(strfind(netbasemodel.params(i).name, '_b'))
        tmp = netbasemodel.params(i).value;
        tmp = zeros(size(tmp), 'single');
        netbasemodel.params(i).value = tmp;
        fprintf('%s \n', netbasemodel.params(i).name);
    elseif ~isempty(strfind(netbasemodel.params(i).name, '_bn')) && ~isempty(strfind(netbasemodel.params(i).name, '_m'))
        tmp = netbasemodel.params(i).value;
        tmp = zeros(size(tmp), 'single');
        netbasemodel.params(i).value = tmp;
        fprintf('%s \n', netbasemodel.params(i).name);
    else
        fprintf('%s ERROR!!!\n', netbasemodel.params(i).name);
    end
end

for i = 1:numel(netbasemodel.params)
    fprintf('%d\t%25s, \t%.2f',i, netbasemodel.params(i).name, netbasemodel.params(i).learningRate);
    fprintf('\tsize: %dx%dx%dx%d\n', size(netbasemodel.params(i).value,1), size(netbasemodel.params(i).value,2), size(netbasemodel.params(i).value,3), size(netbasemodel.params(i).value,4));
end
%% configure training environment
batchSize = 1;
totalEpoch = 100;
learningRate = 1:totalEpoch;
learningRate = (5.0e-3) * (1-learningRate/totalEpoch).^0.9;

weightDecay=0.0005; % weightDecay: usually use the default value

opts.batchSize = batchSize;
opts.learningRate = learningRate;
opts.weightDecay = weightDecay;
opts.momentum = 0.9 ;

opts.expDir = fullfile('./exp', 'main001_instSeg_v1_absEucMM');
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



