clear
% close all
clc;

addpath(genpath('../libs'));
path_to_matconvnet = '/home/skong2/scratch/matconvnet-1.0-beta23_modifiedDagnn';

run(fullfile(path_to_matconvnet, 'matlab', 'vl_setupnn'));
addpath(genpath(fullfile('dependencies', 'matconvnet','examples')));
%% read matconvnet model
load('imdb_toydata_v3_from_mnist.mat');
imdb.path = './toydata_v3';
imdb.path_to_dataset = './toydata_v3';

% set GPU
gpuId = 2; %[1, 2];
gpuDevice(gpuId);
flagSaveFig = true; % {true false} whether to store the result

saveFolder = 'main001_instSeg_v1_absEucMM/';
modelName = 'softmax_net-epoch-31.mat';
%% modify network for testing
netMat = load( fullfile('./exp', saveFolder, modelName) );
netMat = netMat.net;
netMat = dagnn.DagNN.loadobj(netMat);

for loopIdx = 5:-1:1
    rmLayerName = sprintf('loop%d_meanshift_Y_l2norm', loopIdx);
    netMat.vars(netMat.layers(netMat.getLayerIndex(rmLayerName)).outputIndexes).precious = 1;
    
    % add regression loss
    rmLayerName = sprintf('loop%d_instSeg_reg', loopIdx);
    netMat.removeLayer(rmLayerName); % remove layer
    
    % add max-margin loss
    rmLayerName = sprintf('loop%d_instSeg_MM', loopIdx);
    netMat.removeLayer(rmLayerName); % remove layer
    
    % add max-margin loss
    rmLayerName = sprintf('loop%d_meanshift_cosSim', loopIdx);
    netMat.removeLayer(rmLayerName); % remove layer
end

rmLayerName = 'obj_instSeg_MM';
if ~isnan(netMat.getLayerIndex(rmLayerName))
    netMat.removeLayer(rmLayerName); % remove layer
end
rmLayerName = 'obj_instSeg_reg';
if ~isnan(netMat.getLayerIndex(rmLayerName))
    netMat.removeLayer(rmLayerName); % remove layer
end
rmLayerName = 'res7_cosSim';
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

semanticMaskMat = [];
instanceMaskMat = [];
imgMat = [];
predInstanceMaskMat_loop0 = [];
predInstanceMaskMat = {};
for loopIdx = 1:5
    predInstanceMaskMat{loopIdx} = [];
end
st = 0;
for i = 1:5%length(testList)
    curBatch = fullfile(imdb.path_to_dataset, imdb.imgList{testList(i)});
    datamat = load(curBatch);
    
    datamat.semanticMaskMat = reshape(datamat.semanticMaskMat, [size(datamat.semanticMaskMat,1), size(datamat.semanticMaskMat,2), 1, size(datamat.semanticMaskMat,3)]);
    datamat.instanceMaskMat = reshape(datamat.instanceMaskMat, [size(datamat.instanceMaskMat,1), size(datamat.instanceMaskMat,2), 1, size(datamat.instanceMaskMat,3)]);
        
    imFeed = bsxfun(@minus, datamat.imgMat, imdb.meta.meanvalue);
    inputs = {'input', gpuArray(imFeed)};
    netMat.eval(inputs) ;
    res7_l2norm = gather(netMat.vars(netMat.layers(netMat.getLayerIndex('res7_l2norm')).outputIndexes).value);
    predInstanceMaskMat_loop0(:,:,:,st+1:st+size(datamat.imgMat,4)) = res7_l2norm;
    
    for loopIdx = 1:5
        curLayerName = sprintf('loop%d_meanshift_Y_l2norm', loopIdx);
        predInstanceMaskMat{loopIdx}(:,:,:,st+1:st+size(datamat.imgMat,4)) = ...
            gather(netMat.vars(netMat.layers(netMat.getLayerIndex(curLayerName)).outputIndexes).value);
    end

    imgMat(:,:,:,st+1:st+size(datamat.imgMat,4)) = datamat.imgMat;
    semanticMaskMat(:,:,:,st+1:st+size(datamat.imgMat,4)) = datamat.semanticMaskMat;
    instanceMaskMat(:,:,:,st+1:st+size(datamat.imgMat,4)) = datamat.instanceMaskMat;
    %predInstanceMaskMat(:,:,:,st+1:st+size(datamat.imgMat,4)) = res7_l2norm;
    fprintf('\t%d/%d\n', i, length(testList));
    
    st = st + size(datamat.imgMat,4);
end
%% grouping results to visualize
num = 25; % the total number of example to show
M = 5; % the number of rows in the panel, each row shows N images as below
N = 5; % the number of columns in the panel, each column shows M images as above
% h = 28;
% w = 28; 
panelSZ = round(64/2)*2; % the size (height/width) of the small square image
% stepSize = (panelSZ-h)/2;

% image
showcaseSamples = stitchImage2panel(imgMat, panelSZ, M, N, 1, num);

% ground-truth semantic mask
showcaseSemanticMask= semanticMaskMat(:,:,:,1:num);
showcaseSemanticMask = reshape(showcaseSemanticMask, [numel(showcaseSemanticMask(:,:,1,1)), num]);
showcaseSemanticMask = showStitch(showcaseSemanticMask, [panelSZ panelSZ], M, N, 'whitelines', 'linewidth', 5);

% ground-truth instance mask
showcaseInstanceMask= instanceMaskMat(:,:,:,1:num);
showcaseInstanceMask = reshape(showcaseInstanceMask, [numel(showcaseInstanceMask(:,:,1,1)), num]);
showcaseInstanceMask = showStitch(showcaseInstanceMask, [panelSZ panelSZ], M, N, 'whitelines', 'linewidth', 5);


predInstanceMaskMat_loop0 = stitchImage2panel(predInstanceMaskMat_loop0, panelSZ, M, N, 1, num);
predInstanceMaskMat_loop0 = predInstanceMaskMat_loop0 - min(predInstanceMaskMat_loop0(:));
predInstanceMaskMat_loop0 = predInstanceMaskMat_loop0 ./ max(predInstanceMaskMat_loop0(:));

predInstanceMask = {};
for loopIdx = 1:5
    predInstanceMask{loopIdx} = stitchImage2panel(predInstanceMaskMat{loopIdx}, panelSZ, M, N, 1, num);
    predInstanceMask{loopIdx} = predInstanceMask{loopIdx} - min(predInstanceMask{loopIdx}(:));
    predInstanceMask{loopIdx} = predInstanceMask{loopIdx} ./ max(predInstanceMask{loopIdx}(:));
end
%% save the results
figFolder = './figFolder';
if ~isdir(figFolder)
    mkdir(figFolder);
end
a = strfind(modelName,'epoch');
path_to_save = fullfile(figFolder, [strrep(saveFolder,'/','') '_' modelName(a(1):end-4)]);
if ~isdir(path_to_save)
    mkdir(path_to_save);
end

% summary vidualization
imgFig2 = figure(1);
subWindowH = 2;
subWindowW = 2;
set(imgFig2, 'Position', [100 100 1400 900]) % [1 1 width height]
windowID = 1;
subplot(subWindowH, subWindowW, windowID); imagesc(showcaseSamples); axis off image; title('random samples'); colorbar; windowID = windowID + 1;
subplot(subWindowH, subWindowW, windowID); imagesc(showcaseSemanticMask); axis off image; title('showcase SemanticMask'); caxis([0, 11]); colorbar; windowID = windowID + 1;
subplot(subWindowH, subWindowW, windowID); imagesc(showcaseInstanceMask); axis off image; title('showcase instanceMask'); colorbar; windowID = windowID + 1;
subplot(subWindowH, subWindowW, windowID); imagesc(predInstanceMaskMat_loop0); axis off image; title('predInstanceMask'); colorbar; windowID = windowID + 1;
if flagSaveFig
    export_fig( sprintf('%s/fig00_visualization.jpg', path_to_save) );
end


imgFig2 = figure(2);
subWindowH = 2;
subWindowW = 3;
set(imgFig2, 'Position', [100 100 1400 900]) % [1 1 width height]
windowID = 1;
subplot(subWindowH, subWindowW, windowID); imagesc(predInstanceMaskMat_loop0); axis off image; title('loop0'); colorbar; windowID = windowID + 1;
subplot(subWindowH, subWindowW, windowID); imagesc(predInstanceMask{1}); axis off image; title('loop1'); caxis([0, 11]); colorbar; windowID = windowID + 1;
subplot(subWindowH, subWindowW, windowID); imagesc(predInstanceMask{2}); axis off image; title('loop2'); colorbar; windowID = windowID + 1;
subplot(subWindowH, subWindowW, windowID); imagesc(predInstanceMask{3}); axis off image; title('loop3'); colorbar; windowID = windowID + 1;
subplot(subWindowH, subWindowW, windowID); imagesc(predInstanceMask{4}); axis off image; title('loop4'); colorbar; windowID = windowID + 1;
subplot(subWindowH, subWindowW, windowID); imagesc(predInstanceMask{5}); axis off image; title('loop5'); colorbar; windowID = windowID + 1;
if flagSaveFig
    export_fig( sprintf('%s/fig01_visualization_looping.jpg', path_to_save) );
end


if flagSaveFig
    showcaseSemanticMask = uint8(showcaseSemanticMask * 255/11);
    showcaseInstanceMask = showcaseInstanceMask*255/5;
    
    imwrite(showcaseSamples, fullfile(path_to_save, sprintf('fig02_showcaseSamples.bmp')));
    imwrite(showcaseSemanticMask, fullfile(path_to_save, sprintf('fig03_showcaseSemanticMask.bmp')));    
    imwrite(showcaseInstanceMask, fullfile(path_to_save, sprintf('fig04_showcaseInstanceMask.bmp')));
    imwrite(predInstanceMaskMat_loop0, fullfile(path_to_save, sprintf('fig05_predInstanceMask-loop0.bmp')));
    imwrite(predInstanceMask{1}, fullfile(path_to_save, sprintf('fig06_predInstanceMask-loop1.bmp')));
    imwrite(predInstanceMask{2}, fullfile(path_to_save, sprintf('fig07_predInstanceMask-loop2.bmp')));
    imwrite(predInstanceMask{3}, fullfile(path_to_save, sprintf('fig08_predInstanceMask-loop3.bmp')));
    imwrite(predInstanceMask{4}, fullfile(path_to_save, sprintf('fig09_predInstanceMask-loop4.bmp')));
    imwrite(predInstanceMask{5}, fullfile(path_to_save, sprintf('fig10_predInstanceMask-loop5.bmp')));
    
    
    save(sprintf('%s/results.mat', path_to_save), 'imgMat', 'instanceMaskMat', 'semanticMaskMat', 'predInstanceMaskMat');
end

%% leaving blank


