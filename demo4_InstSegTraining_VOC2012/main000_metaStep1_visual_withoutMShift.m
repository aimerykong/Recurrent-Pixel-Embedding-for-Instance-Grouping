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
% netbasemodel.removeLayer('obj_MM_res8');
% netbasemodel.removeLayer('obj_reg_res8');
% netbasemodel.removeLayer('res8_overRes6_cosSim');

outputLayerName = 'output_l2norm';
netbasemodel.vars(netbasemodel.layers(netbasemodel.getLayerIndex(outputLayerName)).outputIndexes).precious = 1;
outputIdx_res7_l2norm = netbasemodel.layers(netbasemodel.getLayerIndex(outputLayerName)).outputIndexes;
%% visualize single image
idxList = 1:length(imdb.val_image);
for idx = idxList    
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
    imgOrg = imresize(imgOrg, reSZ);
    im = bsxfun(@minus, single(imgOrg), imdb.meta.meanvalue) ;
    gtOrg = imresize(gtOrg, reSZ, 'nearest');
    gtDiv8 = imresize(gtOrg, 1/8, 'nearest');
    inputs = {'data', gpuArray(single(im))};
    
    netbasemodel.eval(inputs) ;
    
    feaMap = gather(netbasemodel.vars(outputIdx_res7_l2norm).value);
    feaMapSize = size(feaMap);
    randRGB4Pred = reshape(feaMap, [], size(feaMap,3));
    rng(777); randProj = randn(feaMapSize(3), 3);
    randRGB4Pred = randRGB4Pred*randProj;
    randRGB4Pred = reshape( randRGB4Pred, [feaMapSize(1), feaMapSize(2), 3] );
    randRGB4Pred = rescaleFeaMap(randRGB4Pred);
    
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
    
    imgFig = figure(2);
    set(imgFig, 'Position', [100 100 1500 800]) % [1 1 width height]
    subplot(2,3,1);
    imshow(uint8(imgOrg)); title('original image');
    subplot(2,3,2);
    imagesc(gtOrg); axis off image; title('instance mask');
    subplot(2,3,3);
    imagesc(gtDiv8); axis off image; title('instance mask (div8)')
    subplot(2,3,4);
    imagesc(randRGB4Pred); axis off image; title('rand projection onto 3-dim','Interpreter','none')
    subplot(2,3,5);
    imagesc(randRGB4Pred2); axis off image; title('rand projection onto 3-dim','Interpreter','none')
    subplot(2,3,6);
    imagesc(randRGB4Pred3); axis off image; title('rand projection onto 3-dim','Interpreter','none')
    drawnow;
    
    prefix = sprintf('./figFolder/%s_%s', path_to_model, strrep(modelName,'.mat',''));
    %     if ~isdir(prefix)
    %         mkdir(prefix);
    %     end
    %     export_fig(sprintf('%s/id%d.jpg', prefix, idx));
end

%% leaving blank
%{
%}
