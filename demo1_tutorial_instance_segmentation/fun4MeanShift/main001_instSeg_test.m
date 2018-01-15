clear
% close all
clc;

addpath(genpath('../libs'))
addpath('./myFunc')
% path_to_matconvnet = '../matconvnet';
path_to_matconvnet = '../matconvnet_std';
path_to_matconvnet = '/home/skong2/scratch/matconvnet-1.0-beta23_modifiedDagnn';
path_to_model = '../models/';

run(fullfile(path_to_matconvnet, 'matlab', 'vl_setupnn'));
addpath(genpath(fullfile('dependencies', 'matconvnet','examples')));
%% read matconvnet model
load('imdb_toydata_v3_from_mnist.mat');
% imdb.path_to_dataset = '/run/shm/toydata';

% set GPU
gpuId = 2; %[1, 2];
gpuDevice(gpuId);
flagSaveFig = true; % {true false} whether to store the result


saveFolder = 'main007_instSeg_v1_absEucMM';
modelName = 'softmax_net-epoch-83.mat';

saveFolder = 'main007_instSeg_v2_absEucMM_invAlpha';
modelName = 'softmax_net-epoch-83.mat';

%saveFolder = 'main007_instSeg_v3_absEucMM_adaptMM';
%modelName = 'softmax_net-epoch-92.mat';

%% modify network for testing
netMat = load( fullfile('./exp', saveFolder, modelName) );
netMat = netMat.net;
netMat = dagnn.DagNN.loadobj(netMat);

% RFinfo = netMat.getVarReceptiveFields('input');

rmLayerName = 'obj_instSeg_MM';
if ~isnan(netMat.getLayerIndex(rmLayerName))
    netMat.removeLayer(rmLayerName); % remove layer
end
rmLayerName = 'obj_instSeg';
if ~isnan(netMat.getLayerIndex(rmLayerName))
    netMat.removeLayer(rmLayerName); % remove layer
end
rmLayerName = 'obj_instSeg_reg';
if ~isnan(netMat.getLayerIndex(rmLayerName))
    netMat.removeLayer(rmLayerName); % remove layer
end
netMat.vars(netMat.layers(netMat.getLayerIndex('res7_l2norm')).outputIndexes).precious = 1;
%% configure training environment
saveFolder = [strrep(saveFolder, '/', '') '_visualization'];
if ~isdir(saveFolder) && flagSaveFig
    mkdir(saveFolder);
end
netMat.move('gpu');
% netMat.mode = 'test';
netMat.mode = 'normal';
testList = find(imdb.set==2);
rawCountsAll = {};

imgMat = [];
semanticMaskMat = [];
instanceMaskMat = [];
predInstanceMaskMat = [];
st = 0;
for i = 1:5%length(testList)
    curBatch = fullfile(imdb.path_to_dataset, imdb.imgList{testList(i)});
    datamat = load(curBatch);
    
%     datamat.imgMat = reshape(datamat.imgMat, [size(datamat.imgMat,1), size(datamat.imgMat,2), 1, size(datamat.imgMat,3)]);
    datamat.semanticMaskMat = reshape(datamat.semanticMaskMat, [size(datamat.semanticMaskMat,1), size(datamat.semanticMaskMat,2), 1, size(datamat.semanticMaskMat,3)]);
    datamat.instanceMaskMat = reshape(datamat.instanceMaskMat, [size(datamat.instanceMaskMat,1), size(datamat.instanceMaskMat,2), 1, size(datamat.instanceMaskMat,3)]);
        
    imFeed = bsxfun(@minus, datamat.imgMat, imdb.meta.meanvalue);
    inputs = {'input', gpuArray(imFeed)};
    netMat.eval(inputs) ;
    res7_l2norm = gather(netMat.vars(netMat.layers(netMat.getLayerIndex('res7_l2norm')).outputIndexes).value);

    imgMat(:,:,:,st+1:st+size(datamat.imgMat,4)) = datamat.imgMat;
    semanticMaskMat(:,:,:,st+1:st+size(datamat.imgMat,4)) = datamat.semanticMaskMat;
    instanceMaskMat(:,:,:,st+1:st+size(datamat.imgMat,4)) = datamat.instanceMaskMat;
    predInstanceMaskMat(:,:,:,st+1:st+size(datamat.imgMat,4)) = res7_l2norm;
    fprintf('\t%d/%d\n', i, length(testList));
    
    st = st + size(datamat.imgMat,4);
end
%% visualize
num = 25;
M = 5;
N = 5;
h = 28;
w = 28;
panelSZ = round(64/2)*2;
stepSize = (panelSZ-h)/2;

showcaseSamples_R = imgMat(:,:,1,1:num);
showcaseSamples_R = reshape(showcaseSamples_R, [numel(showcaseSamples_R(:,:,1,1)), num]);
showcaseSamples_R = showStitch(showcaseSamples_R, [panelSZ panelSZ], M, N, 'whitelines', 'linewidth', 5);
showcaseSamples_G = imgMat(:,:,2,1:num);
showcaseSamples_G = reshape(showcaseSamples_G, [numel(showcaseSamples_G(:,:,1,1)), num]);
showcaseSamples_G = showStitch(showcaseSamples_G, [panelSZ panelSZ], M, N, 'whitelines', 'linewidth', 5);
showcaseSamples_B = imgMat(:,:,3,1:num);
showcaseSamples_B = reshape(showcaseSamples_B, [numel(showcaseSamples_B(:,:,1,1)), num]);
showcaseSamples_B = showStitch(showcaseSamples_B, [panelSZ panelSZ], M, N, 'whitelines', 'linewidth', 5);
showcaseSamples = cat(3, showcaseSamples_R, showcaseSamples_G, showcaseSamples_B);


showcaseSemanticMask= semanticMaskMat(:,:,:,1:num);
showcaseSemanticMask = reshape(showcaseSemanticMask, [numel(showcaseSemanticMask(:,:,1,1)), num]);
showcaseSemanticMask = showStitch(showcaseSemanticMask, [panelSZ panelSZ], M, N, 'whitelines', 'linewidth', 5);

showcaseInstanceMask= instanceMaskMat(:,:,:,1:num);
showcaseInstanceMask = reshape(showcaseInstanceMask, [numel(showcaseInstanceMask(:,:,1,1)), num]);
showcaseInstanceMask = showStitch(showcaseInstanceMask, [panelSZ panelSZ], M, N, 'whitelines', 'linewidth', 5);

predInstanceMask_R = predInstanceMaskMat(:,:,1,1:num);
predInstanceMask_R = reshape(predInstanceMask_R, [numel(predInstanceMask_R(:,:,1,1)), num]);
predInstanceMask_R = showStitch(predInstanceMask_R, [panelSZ panelSZ], M, N, 'whitelines', 'linewidth', 5);
predInstanceMask_G = predInstanceMaskMat(:,:,2,1:num);
predInstanceMask_G = reshape(predInstanceMask_G, [numel(predInstanceMask_G(:,:,1,1)), num]);
predInstanceMask_G = showStitch(predInstanceMask_G, [panelSZ panelSZ], M, N, 'whitelines', 'linewidth', 5);
predInstanceMask_B = predInstanceMaskMat(:,:,3,1:num);
predInstanceMask_B = reshape(predInstanceMask_B, [numel(predInstanceMask_B(:,:,1,1)), num]);
predInstanceMask_B = showStitch(predInstanceMask_B, [panelSZ panelSZ], M, N, 'whitelines', 'linewidth', 5);
predInstanceMask = cat(3, predInstanceMask_R, predInstanceMask_G, predInstanceMask_B);

predInstanceMask = predInstanceMask - min(predInstanceMask(:));
predInstanceMask = predInstanceMask ./ max(predInstanceMask(:));

imgFig2 = figure(2);
subWindowH = 2;
subWindowW = 2;
set(imgFig2, 'Position', [100 100 1400 900]) % [1 1 width height]
windowID = 1;
subplot(subWindowH, subWindowW, windowID); imagesc(showcaseSamples); axis off image; title('random samples'); colorbar; windowID = windowID + 1;
subplot(subWindowH, subWindowW, windowID); imagesc(showcaseSemanticMask); axis off image; title('showcase SemanticMask'); caxis([0, 11]); colorbar; windowID = windowID + 1;
subplot(subWindowH, subWindowW, windowID); imagesc(showcaseInstanceMask); axis off image; title('showcase instanceMask'); colorbar; windowID = windowID + 1;
subplot(subWindowH, subWindowW, windowID); imagesc(predInstanceMask); axis off image; title('predInstanceMask'); colorbar; windowID = windowID + 1;

if flagSaveFig
    showcaseSemanticMask = uint8(showcaseSemanticMask * 255/11);
    showcaseInstanceMask = showcaseInstanceMask*255/5;
    
    imwrite(showcaseSamples, fullfile(saveFolder, sprintf('%04d_showcaseSamples.bmp',i)));
    imwrite(showcaseSemanticMask, fullfile(saveFolder, sprintf('%04d_showcaseSemanticMask.bmp',i)));    
    imwrite(showcaseInstanceMask, fullfile(saveFolder, sprintf('%04d_showcaseInstanceMask.bmp',i)));
    imwrite(predInstanceMask, fullfile(saveFolder, sprintf('%04d_predInstanceMask.bmp',i)));
    
    export_fig( sprintf('%s/%04d_visualization.jpg', saveFolder, i) );
    save(sprintf('%s/results.mat', saveFolder), 'imgMat', 'instanceMaskMat', 'semanticMaskMat', 'predInstanceMaskMat');
end

%% leaving blank


