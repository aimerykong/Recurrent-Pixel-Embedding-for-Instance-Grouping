clear
close all
clc;

addpath(genpath('../libs'))
path_to_matconvnet = '/home/skong2/scratch/matconvnet-1.0-beta23_modifiedDagnn';

run(fullfile(path_to_matconvnet, 'matlab', 'vl_setupnn'));
addpath(genpath(fullfile('dependencies', 'matconvnet','examples')));
%% read matconvnet model
load('imdb_toydata_v3_from_mnist.mat');
imdb.path = './toydata_v3';
imdb.path_to_dataset = './toydata_v3';
% set GPU
gpuId = 2; 
gpuDevice(gpuId);
flagSaveFig = true; % {true false} whether to store the result

% saveFolder = 'main007_instSeg_v1_absEucMM';
% modelName = 'softmax_net-epoch-83.mat';

saveFolder = 'main002_instSeg_v1_ftAbsEucMM_epoch83';
modelName = 'softmax_net-epoch-59.mat';



netbasemodel = load( fullfile('./exp', saveFolder, modelName) );
netbasemodel = netbasemodel.net;

% netbasemodel.layers(136).block = rmfield(netbasemodel.layers(136).block, 'ignoreAverage');
% netbasemodel.layers(135).block = rmfield(netbasemodel.layers(135).block, 'ignoreAverage');

netbasemodel = dagnn.DagNN.loadobj(netbasemodel);
%% modify layers, e.g. removing
netbasemodel.meta.normalization.averageImage = imdb.meta.meanvalue; 
netbasemodel.meta.normalization.imageSize = [imdb.meta.height, imdb.meta.width, 1, 1];
%% 1st mean-shift grouping loop
loopNum = 5;

sName_l2norm = 'res7_l2norm';
netbasemodel.vars(netbasemodel.layers(netbasemodel.getLayerIndex(sName_l2norm)).outputIndexes).precious = 1;

% keepLayerName = sprintf('obj_instSeg_reg');
% netbasemodel.layers(netbasemodel.getLayerIndex(keepLayerName)).block.lastLayerName = 'res7_cosSim';
% keepLayerName = sprintf('obj_instSeg_MM');
% netbasemodel.layers(netbasemodel.getLayerIndex(keepLayerName)).block.lastLayerName = 'res7_cosSim';

rmLayerName = 'obj_instSeg_MM';
if ~isnan(netbasemodel.getLayerIndex(rmLayerName))
    netbasemodel.removeLayer(rmLayerName); % remove layer
end
rmLayerName = 'obj_instSeg_reg';
if ~isnan(netbasemodel.getLayerIndex(rmLayerName))
    netbasemodel.removeLayer(rmLayerName); % remove layer
end
rmLayerName = 'res7_cosSim';
if ~isnan(netbasemodel.getLayerIndex(rmLayerName))
    netbasemodel.removeLayer(rmLayerName); % remove layer
end

gt_name =  sprintf('gt_ins');
weight_for_losses = {'obj_instSeg_reg', 1, 'obj_instSeg_MM', 1};
for loopIdx = 1:loopNum
%     [netbasemodel, sName, sName_l2norm] = addOneLoop_forMeanShiftGrouping(netbasemodel, sName_l2norm, loopIdx);
    
    keepLayerName = sprintf('loop%d_meanshift_S_is_XX', loopIdx);
    netbasemodel.layers(netbasemodel.getLayerIndex(keepLayerName)).block.analyzeGradient = false;
    
    keepLayerName = sprintf('loop%d_meanshift_Y_l2norm', loopIdx);
    netbasemodel.vars(netbasemodel.layers(netbasemodel.getLayerIndex(keepLayerName)).outputIndexes).precious = 1;  
    
    rmLayerName = sprintf('loop%d_meanshift_cosSim', loopIdx);
    netbasemodel.removeLayer(rmLayerName); % remove layer
    
    rmLayerName = sprintf('loop%d_instSeg_reg', loopIdx);
    netbasemodel.removeLayer(rmLayerName); % remove layer
    
    rmLayerName = sprintf('loop%d_instSeg_MM', loopIdx);
    netbasemodel.removeLayer(rmLayerName); % remove layer
end
%%

netbasemodel.move('gpu');
% netMat.mode = 'test';
netbasemodel.mode = 'normal';
testList = find(imdb.set==2);
rawCountsAll = {};

semanticMaskMat = [];
instanceMaskMat = [];
imgMat = [];
predInstanceMaskMat_loop0 = [];
predInstanceMaskMat = {};
for loopIdx = 1:loopNum
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
    netbasemodel.eval(inputs) ;
    res7_l2norm = gather(netbasemodel.vars(netbasemodel.layers(netbasemodel.getLayerIndex('res7_l2norm')).outputIndexes).value);
    predInstanceMaskMat_loop0(:,:,:,st+1:st+size(datamat.imgMat,4)) = res7_l2norm;
    
    for loopIdx = 1:loopNum
        curLayerName = sprintf('loop%d_meanshift_Y_l2norm', loopIdx);
        predInstanceMaskMat{loopIdx}(:,:,:,st+1:st+size(datamat.imgMat,4)) = ...
            gather(netbasemodel.vars(netbasemodel.layers(netbasemodel.getLayerIndex(curLayerName)).outputIndexes).value);
    end

    imgMat(:,:,:,st+1:st+size(datamat.imgMat,4)) = datamat.imgMat;
    semanticMaskMat(:,:,:,st+1:st+size(datamat.imgMat,4)) = datamat.semanticMaskMat;
    instanceMaskMat(:,:,:,st+1:st+size(datamat.imgMat,4)) = datamat.instanceMaskMat;
    fprintf('\t%d/%d\n', i, length(testList));
    
    st = st + size(datamat.imgMat,4);
end
predInstanceMaskMat0 = predInstanceMaskMat_loop0;
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
for loopIdx = 1:loopNum
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
for loopIdx = 1:5
    subplot(subWindowH, subWindowW, windowID); 
    imagesc(predInstanceMask{loopIdx}); axis off image; title(sprintf('loop%d',loopIdx)); caxis([0, 11]); colorbar;    
    windowID = windowID + 1;
end
if flagSaveFig
    export_fig( sprintf('%s/fig01_visualization_looping1.jpg', path_to_save) );
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
%{
%}



