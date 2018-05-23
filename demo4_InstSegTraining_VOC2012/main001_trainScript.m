clear; clc; close all;

addpath('../libs/exportFig');
addpath('../libs/fun4MeanShift');
addpath('../libs/layerExt');
addpath('../libs/myFunctions/');
path_to_matconvnet = '../libs/matconvnet-1.0-beta23_modifiedDagnn';
run(fullfile(path_to_matconvnet, 'matlab', 'vl_setupnn'));
addpath(genpath(fullfile('dependencies', 'matconvnet','examples')));

% set GPU
gpuId = 3;
gpuDevice(gpuId);
%% load imdb file
load('imdb_complete_on_server.mat');
imdb.meta.meanvalue = reshape(imdb.meta.meanvalue,[1 1 3]);
imdb.path_to_image = '/home/skong2/dataset/';
imdb.path_to_annotation = '/home/skong2/dataset/';
%% sepcify model
path_to_model = 'main001_ft4COCO_baseline_v1';
modelName = 'pairMMAbsReg_net-epoch-1.mat';

netbasemodel = load( fullfile('./exp', path_to_model, modelName) );
netbasemodel = netbasemodel.net;
netbasemodel = dagnn.DagNN.loadobj(netbasemodel);
% add objective function layer
scalingFactor = 4;
netbasemodel.meta.normalization.averageImage = reshape([123.68, 116.779,  103.939],[1,1,3]); % imagenet mean values
%% Insert new layers.
netbasemodel.removeLayer('obj_MM'); % remove layer
netbasemodel.removeLayer('obj_reg'); % remove layer
netbasemodel.removeLayer('output_cosSim'); % remove layer

sName = 'output_l2norm';
lName = 'output_cosSim';
gt_name =  sprintf('gt_ins_div4');
netbasemodel.addLayer(lName, cosineSimilarity_randSample('randSampleRatio', 0.35), {sName, gt_name}, lName) ;
sName = lName;

% add regression loss
obj_name = sprintf('obj_reg');
netbasemodel.addLayer(obj_name, ...
    InstanceSegRegLoss_randSample('loss', 'cosinesimilarityabsregloss', 'lastLayerName', sName), ... softmaxlog logistic
    {sName, gt_name}, obj_name);

% add max-margin loss
obj_name = sprintf('obj_MM');
input_name = sName;
netbasemodel.addLayer(obj_name, ...
    InstanceSegMMLoss_randSample('loss', 'cosinesimilaritymmloss', ...
    'marginAlpha_', 0.5, ...
    'adaptiveMM', false, ...
    'lastLayerName', sName), ...
    {input_name, gt_name}, obj_name)

%% set learning rate for layers
for i = 1:numel(netbasemodel.params)
    if contains(netbasemodel.params(i).name,'bn_m')
        netbasemodel.params(i).learningRate = 0.1;
    else
        netbasemodel.params(i).learningRate = 1;
    end
end

ind = netbasemodel.layers(netbasemodel.getLayerIndex('res7_interp')).paramIndexes;
netbasemodel.params(ind).learningRate = 0;

for i = 1:numel(netbasemodel.params)
    fprintf('%d \t%s, \t\t%.2f\n',i, netbasemodel.params(i).name, netbasemodel.params(i).learningRate);
end
%% configure training environment
batchSize = 1;
totalEpoch = 150;
learningRate = 1:totalEpoch;
learningRate = (1e-5) * (1-learningRate/totalEpoch).^0.9;
weightDecay=0.0005; % weightDecay: usually use the default value

opts.batchSize = batchSize;
opts.learningRate = learningRate;
opts.weightDecay = weightDecay;
opts.momentum = 0.9 ;

opts.scalingFactor = scalingFactor;

opts.expDir = fullfile('./exp', 'main001_ft4COCO_baseline_v1');
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

% opts.train = imdb.voc2012_trainaug;
% opts.val = imdb.voc2012_val;
opts.train = 1:length(imdb.train_image);
opts.val = 1:length(imdb.val_image);

opts.checkpointFn = [];
mopts.classifyType = 'pairMMAbsReg';

rng(777);
bopts = netbasemodel.meta.normalization;
bopts.numThreads = 12;
bopts.imdb = imdb;
%% train
fn = getBatchWrapper_augVOC2012(bopts);

opts.backPropDepth = inf; % could limit the backprop
prefixStr = [mopts.classifyType, '_'];
opts.backPropAboveLayerName = 'res4_1_projBranch'; % for fast fine-tuning by freezing all layers below this one
%opts.backPropAboveLayerName = 'conv1_1';

trainfn = @cnnTrain_augVOC2012_inst_div4;
[netbasemodel, info] = trainfn(netbasemodel, prefixStr, imdb, fn, ...
    'derOutputs', {'obj_reg', 1, 'obj_MM', 5}, opts);

%% leaving blank
