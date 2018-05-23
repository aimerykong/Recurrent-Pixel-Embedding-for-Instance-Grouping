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
%% read matconvnet model
load('imdb_complete_on_server.mat');
imdb.meta.meanvalue = reshape(imdb.meta.meanvalue,[1 1 3]);
imdb.path_to_image = '/home/skong2/dataset/';
imdb.path_to_annotation = '/home/skong2/dataset/';

path_to_model = 'main001_ft4COCO_baseline_v1';
modelName = 'pairMMAbsReg_net-epoch-1.mat'; 

netbasemodel = load( fullfile('./exp', path_to_model, modelName) );
netbasemodel = netbasemodel.net;
netbasemodel = dagnn.DagNN.loadobj(netbasemodel);
netbasemodel.move('gpu') ;
netbasemodel.mode = 'normal'; % {test, normal} !! normal --> per-instance normalization
netbasemodel.conserveMemory = 1;

netbasemodel.removeLayer('obj_MM');
netbasemodel.removeLayer('obj_reg');
netbasemodel.removeLayer('output_cosSim');
netbasemodel.removeLayer('output_l2norm');

sName = 'res9_conv';
res9_conv = netbasemodel.layers(netbasemodel.getLayerIndex(sName)).outputIndexes;
netbasemodel.vars(res9_conv).precious = 1;

lName = 'output_l2norm';
netbasemodel.addLayer(lName, L2normalization(), sName, lName) ;
sName = lName;
outputIdx_l2norm = netbasemodel.layers(netbasemodel.getLayerIndex('output_l2norm')).outputIndexes;
netbasemodel.vars(outputIdx_l2norm).precious = 1;

loopNum = 16;
predInstanceMaskMat_loop0 = cell(1,3);
predInstanceMaskMat_loops = cell(loopNum,3);

% alpha = 0.5;
% GaussianBandwidth = 1-alpha;
GaussianBandwidth = 0.2;
    
sName_l2norm = lName;
for loopIdx = 1:loopNum
    [netbasemodel, sName, sName_l2norm] = addOneLoop_forMeanShiftGrouping(netbasemodel, sName_l2norm, loopIdx, GaussianBandwidth);
    
    keepLayerName = sprintf('loop%d_meanshift_S_is_XX', loopIdx);
    netbasemodel.layers(netbasemodel.getLayerIndex(keepLayerName)).block.analyzeGradient = false;
    
    keepLayerName = sprintf('loop%d_meanshift_Y_l2norm', loopIdx);
    netbasemodel.vars(netbasemodel.layers(netbasemodel.getLayerIndex(keepLayerName)).outputIndexes).precious = 1;
    
    rmLayerName = sprintf('loop%d_meanshift_cosSim', loopIdx);
    netbasemodel.removeLayer(rmLayerName); % remove layer
end
%% visualize single image
idxList = 1:length(imdb.val_image);
for idx = idxList
    %% read image and annotation
    cur_path_to_image = fullfile(imdb.path_to_image, imdb.val_image{idx});
    cur_path_to_annot = fullfile(imdb.path_to_annot, imdb.val_annot{idx});
    cur_path_to_image = strrep(cur_path_to_image,'\','/');
    cur_path_to_annot = strrep(cur_path_to_annot,'\','/');
    
    imgOrg = imread(cur_path_to_image);
    cur_segMap = load(cur_path_to_annot);
    cur_instMap = cur_segMap.gtMat.instMap;
    cur_segMap = cur_segMap.gtMat.segMap;
    gtOrg = cur_instMap;
    
    sz = size(gtOrg);
    reSZ = round(sz/8)*8;
    img = imresize(imgOrg, reSZ);
    
    instMap_div8 = imresize(cur_instMap, reSZ/8, 'nearest');    
    img = bsxfun(@minus, single(img), imdb.meta.meanvalue) ;
    inputs = {'data', gpuArray(single(img))};    
    netbasemodel.eval(inputs) ;    
    %% gather result
    keepLayerName = 'output_l2norm';
    keepLayerName = netbasemodel.layers(netbasemodel.getLayerIndex(keepLayerName)).outputIndexes;
    feaMap = gather(netbasemodel.vars(keepLayerName).value);
    
    feaMapSize = size(feaMap);
    randRGB4Pred1 = reshape(feaMap, [], size(feaMap,3));
    rng(777); randProj = randn(feaMapSize(3), 3);
    randRGB4Pred1 = randRGB4Pred1*randProj;
    randRGB4Pred1 = reshape( randRGB4Pred1, [feaMapSize(1), feaMapSize(2), 3] );
    randRGB4Pred1 = rescaleFeaMap(randRGB4Pred1);
    
    rng(77); randProj = randn(feaMapSize(3), 3);
    randRGB4Pred2 = reshape(feaMap, [], size(feaMap,3));
    randRGB4Pred2 = randRGB4Pred2*randProj;
    randRGB4Pred2 = reshape( randRGB4Pred2, [feaMapSize(1), feaMapSize(2), 3] );
    randRGB4Pred2 = rescaleFeaMap(randRGB4Pred2);
    
    rng(7); randProj = randn(feaMapSize(3), 3);
    randRGB4Pred3 = reshape(feaMap, [], size(feaMap,3));
    randRGB4Pred3 = randRGB4Pred3*randProj;
    randRGB4Pred3 = reshape( randRGB4Pred3, [feaMapSize(1), feaMapSize(2), 3] );
    randRGB4Pred3 = rescaleFeaMap(randRGB4Pred3);
    
    randRGB4Pred1 = imresize(randRGB4Pred1, [size(imgOrg,1),size(imgOrg,2)]);
    randRGB4Pred2 = imresize(randRGB4Pred2, [size(imgOrg,1),size(imgOrg,2)]);
    randRGB4Pred3 = imresize(randRGB4Pred3, [size(imgOrg,1),size(imgOrg,2)]);

    %     imwrite(uint8(255*randRGB4Pred1), sprintf('%s/valId%04d/randRGB4Pred1.png', ...
    %         prefix, idx));
    %     imwrite(uint8(255*randRGB4Pred2), sprintf('%s/valId%04d/randRGB4Pred2.png', ...
    %         prefix, idx));
    %     imwrite(uint8(255*randRGB4Pred3), sprintf('%s/valId%04d/randRGB4Pred3.png', ...
    %         prefix, idx));
    %% after some loops of mean shift
    keepLayerName = sprintf('loop%d_meanshift_Y_l2norm', loopNum);
    keepLayerName = netbasemodel.layers(netbasemodel.getLayerIndex(keepLayerName)).outputIndexes;
    feaMap = gather(netbasemodel.vars(keepLayerName).value);
    
    feaMapSize = size(feaMap);
    randRGB4Pred1_afterMShift = reshape(feaMap, [], size(feaMap,3));
    rng(777); randProj = randn(feaMapSize(3), 3);
    randRGB4Pred1_afterMShift = randRGB4Pred1_afterMShift*randProj;
    randRGB4Pred1_afterMShift = reshape( randRGB4Pred1_afterMShift, [feaMapSize(1), feaMapSize(2), 3] );
    randRGB4Pred1_afterMShift = rescaleFeaMap(randRGB4Pred1_afterMShift);
    
    rng(77); randProj = randn(feaMapSize(3), 3);
    randRGB4Pred2_afterMShift = reshape(feaMap, [], size(feaMap,3));
    randRGB4Pred2_afterMShift = randRGB4Pred2_afterMShift*randProj;
    randRGB4Pred2_afterMShift = reshape( randRGB4Pred2_afterMShift, [feaMapSize(1), feaMapSize(2), 3] );
    randRGB4Pred2_afterMShift = rescaleFeaMap(randRGB4Pred2_afterMShift);
    
    rng(7); randProj = randn(feaMapSize(3), 3);
    randRGB4Pred3_afterMShift = reshape(feaMap, [], size(feaMap,3));
    randRGB4Pred3_afterMShift = randRGB4Pred3_afterMShift*randProj;
    randRGB4Pred3_afterMShift = reshape( randRGB4Pred3_afterMShift, [feaMapSize(1), feaMapSize(2), 3] );
    randRGB4Pred3_afterMShift = rescaleFeaMap(randRGB4Pred3_afterMShift);
    
    randRGB4Pred1_afterMShift = imresize(randRGB4Pred1_afterMShift, [size(imgOrg,1),size(imgOrg,2)]);
    randRGB4Pred2_afterMShift = imresize(randRGB4Pred2_afterMShift, [size(imgOrg,1),size(imgOrg,2)]);
    randRGB4Pred3_afterMShift = imresize(randRGB4Pred3_afterMShift, [size(imgOrg,1),size(imgOrg,2)]);
    
    figHandler = figure(1);
    set(figHandler, 'Position', [100 100 1500 800]) % [1 1 width height]
    subplot(3,3,1); imshow(imgOrg); title('imgOrg');
    subplot(3,3,2); imagesc(cur_segMap); axis off image; title('segMap');
    subplot(3,3,3); imagesc(cur_instMap); axis off image; title('instMap');
    
    subplot(3,3,4); imagesc(randRGB4Pred1); axis off image; title('randRGB4Pred1');
    subplot(3,3,5); imagesc(randRGB4Pred2); axis off image; title('randRGB4Pred2');
    subplot(3,3,6); imagesc(randRGB4Pred3); axis off image; title('randRGB4Pred3');
    subplot(3,3,7); imagesc(randRGB4Pred1_afterMShift); axis off image; title('randRGB4Pred1_afterMShift','Interpreter','none');
    subplot(3,3,8); imagesc(randRGB4Pred2_afterMShift); axis off image; title('randRGB4Pred2_afterMShift','Interpreter','none');
    subplot(3,3,9); imagesc(randRGB4Pred3_afterMShift); axis off image; title('randRGB4Pred3_afterMShift','Interpreter','none');
    drawnow;
end
%% leaving blank
%{
%}
