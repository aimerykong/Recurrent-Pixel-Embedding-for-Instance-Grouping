clear; clc; close all;
addpath '../libs/exportFig/';
addpath '../libs/layerExt/';
addpath '../libs/myFunctions/';
path_to_matconvnet = '../libs/matconvnet-1.0-beta23_modifiedDagnn';
run(fullfile(path_to_matconvnet, 'matlab', 'vl_setupnn'));
addpath(genpath(fullfile('dependencies', 'matconvnet','examples')));

path_to_saveFigure = './figures';
if ~isdir(path_to_saveFigure)
    mkdir(path_to_saveFigure);
end
%% read matconvnet model
load('imdb.mat');
imdb.meta.meanvalue = reshape(imdb.meta.meanvalue,[1 1 3]);
% set GPU
gpuId = 1;
gpuDevice(gpuId);

netbasemodel = load(fullfile('models/main014_binaryV11_over3dim_v1_reweight_fuse',...
    'pairMMAbsReg_net-epoch-387.mat'));
netbasemodel = netbasemodel.net;
for i = 1:length(netbasemodel.layers)
    if isfield(netbasemodel.layers(i).block, 'bnorm_moment_type_trn')
        netbasemodel.layers(i).block = rmfield(netbasemodel.layers(i).block, 'noise_param_idx');
        netbasemodel.layers(i).block = rmfield(netbasemodel.layers(i).block, 'noise_cache_size');
        netbasemodel.layers(i).block = rmfield(netbasemodel.layers(i).block, 'bnorm_moment_type_trn');
        netbasemodel.layers(i).block = rmfield(netbasemodel.layers(i).block, 'bnorm_moment_type_tst');
    end
end
netbasemodel = dagnn.DagNN.loadobj(netbasemodel);
netbasemodel.move('gpu') ;

netbasemodel.removeLayer('obj_softmax_concat');
netbasemodel.removeLayer('obj_softmax_res5');
netbasemodel.removeLayer('obj_softmax_res4');
netbasemodel.removeLayer('obj_softmax_res3');
netbasemodel.removeLayer('obj_softmax_res2');

outputIdx_res2 = netbasemodel.layers(netbasemodel.getLayerIndex('res7_overRes2_l2norm')).outputIndexes;
netbasemodel.vars(outputIdx_res2).precious = 1;%  res7_l2norm_overRes2
outputIdx_res3 = netbasemodel.layers(netbasemodel.getLayerIndex('res7_overRes3_l2norm')).outputIndexes;
netbasemodel.vars(outputIdx_res3).precious = 1;%  res7_l2norm_overRes3
outputIdx_res4 = netbasemodel.layers(netbasemodel.getLayerIndex('res7_overRes4_l2norm')).outputIndexes;
netbasemodel.vars(outputIdx_res4).precious = 1;%  res7_l2norm_overRes4
outputIdx_res5 = netbasemodel.layers(netbasemodel.getLayerIndex('res7_overRes5_l2norm')).outputIndexes;
netbasemodel.vars(outputIdx_res5).precious = 1;%  res7_l2norm_overRes5
idx_binary_res2 = netbasemodel.layers(netbasemodel.getLayerIndex('res8_conv_overRes2')).outputIndexes;
netbasemodel.vars(idx_binary_res2).precious = 1;%
idx_binary_res3 = netbasemodel.layers(netbasemodel.getLayerIndex('res8_conv_overRes3')).outputIndexes;
netbasemodel.vars(idx_binary_res3).precious = 1;%
idx_binary_res4 = netbasemodel.layers(netbasemodel.getLayerIndex('res8_conv_overRes4')).outputIndexes;
netbasemodel.vars(idx_binary_res4).precious = 1;%
idx_binary_res5 = netbasemodel.layers(netbasemodel.getLayerIndex('res8_conv_overRes5')).outputIndexes;
netbasemodel.vars(idx_binary_res5).precious = 1;%
idx_binary_concat= netbasemodel.layers(netbasemodel.getLayerIndex('res8_conv_overConcat')).outputIndexes;
netbasemodel.vars(idx_binary_concat).precious = 1;%

% netbasemodel.conserveMemory = 0;
netbasemodel.mode = 'test' ;
%% visualize single image
imgList = dir('./data/*mat');
for idx = 1:length(imgList)
    % curMat = load(fullfile(imdb.path_to_dataset, imdb.dataList{testList(idx)}));
    curMat = load(fullfile('./data', imgList(idx).name));
    imgOrg = single(curMat.im); 
    gtOrg = single(curMat.GT);
    sz = size(gtOrg); reSZ = round(sz/8)*4;
    imgOrg = imresize(imgOrg, reSZ); im = bsxfun(@minus, imgOrg, imdb.meta.meanvalue) ;
    inputs = {'data', gpuArray(single(im))};%, 'gt_ins', gpuArray(gtDiv8)};
    netbasemodel.eval(inputs) ;
        
    feaMap_res2 = gather(netbasemodel.vars(outputIdx_res2).value);
    feaMap_res3 = gather(netbasemodel.vars(outputIdx_res3).value);
    feaMap_res4 = gather(netbasemodel.vars(outputIdx_res4).value);
    feaMap_res5 = gather(netbasemodel.vars(outputIdx_res5).value);
    feaMap_res2 = rescaleFeaMap(feaMap_res2);
    feaMap_res3 = rescaleFeaMap(feaMap_res3);
    feaMap_res4 = rescaleFeaMap(feaMap_res4);
    feaMap_res5 = rescaleFeaMap(feaMap_res5);
    
    pred_res2_conf = gather(netbasemodel.vars(idx_binary_res2).value);
    pred_res3_conf = gather(netbasemodel.vars(idx_binary_res3).value);
    pred_res4_conf = gather(netbasemodel.vars(idx_binary_res4).value);
    pred_res5_conf = gather(netbasemodel.vars(idx_binary_res5).value);
    pred_concat_conf = gather(netbasemodel.vars(idx_binary_concat).value);
    
    [~, pred_res2] = max(pred_res2_conf,[], 3);
    [~, pred_res3] = max(pred_res3_conf,[], 3);
    [~, pred_res4] = max(pred_res4_conf,[], 3);
    [~, pred_res5] = max(pred_res5_conf,[], 3);
    [~, pred_concat] = max(pred_concat_conf,[], 3);
    
    
    imgFig = figure;
    set(imgFig, 'Position', [100 100 1500 800]) % [1 1 width height]
    winH = 3;
    winW = 5;
    winIdx = 1;
    subplot(winH, winW, winIdx); winIdx = winIdx + 1; 
    imshow(uint8(imgOrg)); title('original image');    
    subplot(winH, winW, winIdx); winIdx = winIdx + winW - 1; 
    imagesc(gtOrg); axis off image; title('annotation');
    subplot(winH, winW, winIdx); winIdx = winIdx + 1; 
    imagesc(feaMap_res2); axis off image; title('grouping @ ResBlock2');
    subplot(winH, winW, winIdx); winIdx = winIdx + 1; 
    imagesc(feaMap_res3); axis off image; title('grouping @ ResBlock3');
    subplot(winH, winW, winIdx); winIdx = winIdx + 1; 
    imagesc(feaMap_res4); axis off image; title('grouping @ ResBlock4');
    subplot(winH, winW, winIdx); winIdx = winIdx + 1; 
    imagesc(feaMap_res5); axis off image; title('grouping @ ResBlock5');
    
    feaMapMerge = cat(3, feaMap_res2, feaMap_res3, feaMap_res4, feaMap_res5);
    feaMapSize = size(feaMapMerge);
    feaMapMerge = reshape(feaMapMerge, [], size(feaMapMerge,3));
    rng(7); randProj = randn(feaMapSize(3), 3);
    feaMapMerge = feaMapMerge*randProj;
    feaMapMerge = reshape( feaMapMerge, [feaMapSize(1), feaMapSize(2), 3] );
    feaMapMerge = rescaleFeaMap(feaMapMerge);
    subplot(winH, winW, winIdx); winIdx = winIdx + 1; 
    imagesc(feaMapMerge); axis off image; caxis([1,2]); title('randProj to merge');  
            
    
    subplot(winH, winW, winIdx); winIdx = winIdx + 1; 
    imagesc(pred_res2); axis off image; caxis([1,2]); title('binary pred @ res2');    
    subplot(winH, winW, winIdx); winIdx = winIdx + 1; 
    imagesc(pred_res3); axis off image; caxis([1,2]); title('binary pred @ res3');    
    subplot(winH, winW, winIdx); winIdx = winIdx + 1; 
    imagesc(pred_res4); axis off image; caxis([1,2]); title('binary pred @ res4');    
    subplot(winH, winW, winIdx); winIdx = winIdx + 1; 
    imagesc(pred_res5); axis off image; caxis([1,2]); title('binary pred @ res5');
    subplot(winH, winW, winIdx); winIdx = winIdx + 1; 
    imagesc(pred_concat); axis off image; caxis([1,2]); title('binary pred @ fuse');
    
    hierarchMap = single(pred_concat_conf) + single(pred_res5_conf) + single(pred_res4_conf) + single(pred_res3_conf) + single(pred_res2_conf);
    hierarchMap = hierarchMap ./ 5;   
    hierarchMap = hierarchMap(:,:,2);
    subplot(winH, winW, 3);
    imshow(hierarchMap); title('hierarchical boundary');
    %% final output after thinning
    r = 4; 
    s = 1;
    f = [1:r r+1 r:-1:1]/(r+1)^2;
    J = padarray(hierarchMap, [r r], 'symmetric','both');
    J = convn(convn(J,f,'valid'),f','valid');
    if(s>1)
        t=floor(s/2)+1; 
        J = J(t:s:end-s+t,t:s:end-s+t,:); 
    end   

    [Ox,Oy] = gradient(J);
    [Oxx,~] = gradient(Ox);
    [Oxy,Oyy] = gradient(Oy);
    O = mod(atan(Oyy.*sign(-Oxy)./(Oxx+1e-5)),pi);
    
    E = edgesNmsMex(hierarchMap, O, 1, 5, 1.01, 4);
    
    subplot(winH, winW, 4);
    imshow(E); title('thin boundary');
    %% save
    [~,savename] = fileparts(imgList(idx).name);
    export_fig(sprintf('./%s/%s.jpg', path_to_saveFigure, savename));
end
%% leaving blank


