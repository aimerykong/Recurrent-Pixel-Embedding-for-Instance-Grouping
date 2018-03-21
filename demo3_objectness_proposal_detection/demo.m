clear; clc;

addpath(genpath('../libs/exportFig'));
addpath(genpath('../libs/fun4MeanShift'));
addpath(genpath('../libs/layerExt'));
addpath(genpath('../libs/layerExt'));
path_to_matconvnet = '../libs/matconvnet-1.0-beta23_modifiedDagnn';
run(fullfile(path_to_matconvnet, 'matlab', 'vl_setupnn'));
addpath(genpath(fullfile('dependencies', 'matconvnet','examples')));
%% read matconvnet model
load('imdb_complete_on_server.mat');
imdb.meta.meanvalue = reshape(imdb.meta.meanvalue,[1 1 3]);

% set GPU
gpuId = 1;
gpuDevice(gpuId);

modelName = 'basemodel_resnet101_div4_loopNum0.mat';
netbasemodel = load( fullfile('./basemodel', modelName) );
netbasemodel = dagnn.DagNN.loadobj(netbasemodel);
netbasemodel.move('gpu') ;

netbasemodel.removeLayer('obj_MM');
netbasemodel.removeLayer('obj_reg');
netbasemodel.removeLayer('output_cosSim');
netbasemodel.removeLayer('output_l2norm');

%netbasemodel.mode = 'test' ;
netbasemodel.mode = 'normal' ;
net.conserveMemory = 1;
sName = 'res9_conv';

lName = 'output_l2norm';
netbasemodel.addLayer(lName, L2normalization(), sName, lName) ;
sName = lName;
outputIdx_l2norm = netbasemodel.layers(netbasemodel.getLayerIndex('output_l2norm')).outputIndexes;
netbasemodel.vars(outputIdx_l2norm).precious = 1;

loopNum = 16; % let it loop and visualize the intermediate results
predInstanceMaskMat_loop0 = cell(1,3);
predInstanceMaskMat_loops = cell(loopNum,3);

sName_l2norm = lName;
for loopIdx = 1:loopNum
    [netbasemodel, sName, sName_l2norm] = addOneLoop_forMeanShiftGrouping(netbasemodel, sName_l2norm, loopIdx);

    keepLayerName = sprintf('loop%d_meanshift_S_is_XX', loopIdx);
    netbasemodel.layers(netbasemodel.getLayerIndex(keepLayerName)).block.analyzeGradient = false;

    keepLayerName = sprintf('loop%d_meanshift_Y_l2norm', loopIdx);
    netbasemodel.vars(netbasemodel.layers(netbasemodel.getLayerIndex(keepLayerName)).outputIndexes).precious = 1;

    rmLayerName = sprintf('loop%d_meanshift_cosSim', loopIdx);
    netbasemodel.removeLayer(rmLayerName); % remove layer
end
%% visualize single image
prefix4saving = './results/'; % all visual results are saved here
if ~isdir(prefix4saving)
    mkdir(prefix4saving);
end
path_to_image = './images/';
imgList = dir(fullfile(path_to_image, '*jpg'));
for idx = 1:length(imgList)
    fprintf('%d %s\n', idx, imgList(idx).name)
    cur_path_to_image = fullfile(path_to_image, imgList(idx).name);
    imgOrg = imread(cur_path_to_image);
    
    [~,curFileName,curFileExt] = fileparts(cur_path_to_image); 
    cur_path_to_annot = fullfile(path_to_image, [curFileName,'.mat']);
    cur_segMap = load(cur_path_to_annot);
    cur_instMap = cur_segMap.gtMat.instMap;
    cur_segMap = cur_segMap.gtMat.segMap;    
    gtOrg = cur_instMap;
    
    sz = size(gtOrg);
    reSZ = round(sz/8)*8;
    
    img = imresize(imgOrg, reSZ);
    im = bsxfun(@minus, single(img), imdb.meta.meanvalue) ;
    inputs = {'data', gpuArray(single(im))};
    
    netbasemodel.eval(inputs) ;
    %% gather result
    feaMap = gather(netbasemodel.vars(outputIdx_l2norm).value);
    feaMapSize = size(feaMap);
    predInstanceMaskMat_loop0{1} = reshape(feaMap, [], size(feaMap,3));
    rng(777); randProj = randn(feaMapSize(3), 3);
    predInstanceMaskMat_loop0{1} = predInstanceMaskMat_loop0{1}*randProj;
    predInstanceMaskMat_loop0{1} = reshape( predInstanceMaskMat_loop0{1}, [feaMapSize(1), feaMapSize(2), 3] );
    predInstanceMaskMat_loop0{1} = rescaleFeaMap(predInstanceMaskMat_loop0{1});
    predInstanceMaskMat_loop0{1} = imresize(predInstanceMaskMat_loop0{1}, [sz(1),sz(2)]);
    
    rng(77); randProj = randn(feaMapSize(3), 3);
    predInstanceMaskMat_loop0{2} = reshape(feaMap, [], size(feaMap,3));
    predInstanceMaskMat_loop0{2} = predInstanceMaskMat_loop0{2}*randProj;
    predInstanceMaskMat_loop0{2} = reshape( predInstanceMaskMat_loop0{2}, [feaMapSize(1), feaMapSize(2), 3] );
    predInstanceMaskMat_loop0{2} = rescaleFeaMap(predInstanceMaskMat_loop0{2});
    predInstanceMaskMat_loop0{2} = imresize(predInstanceMaskMat_loop0{2}, [sz(1),sz(2)]);
    
    rng(7); randProj = randn(feaMapSize(3), 3);
    predInstanceMaskMat_loop0{3} = reshape(feaMap, [], size(feaMap,3));
    predInstanceMaskMat_loop0{3} = predInstanceMaskMat_loop0{3}*randProj;
    predInstanceMaskMat_loop0{3} = reshape( predInstanceMaskMat_loop0{3}, [feaMapSize(1), feaMapSize(2), 3] );
    predInstanceMaskMat_loop0{3} = rescaleFeaMap(predInstanceMaskMat_loop0{3});
    predInstanceMaskMat_loop0{3} = imresize(predInstanceMaskMat_loop0{3}, [sz(1),sz(2)]);
    
    for loopidx = 1:loopNum
        keepLayerName = sprintf('loop%d_meanshift_Y_l2norm', loopIdx);
        keepLayerName = netbasemodel.layers(netbasemodel.getLayerIndex(keepLayerName)).outputIndexes;
        feaMap = gather(netbasemodel.vars(keepLayerName).value);
        feaMapSize = size(feaMap);
        predInstanceMaskMat_loops{loopidx,1} = reshape(feaMap, [], size(feaMap,3));
        rng(777); randProj = randn(feaMapSize(3), 3);
        predInstanceMaskMat_loops{loopidx,1} = predInstanceMaskMat_loops{loopidx,1}*randProj;
        predInstanceMaskMat_loops{loopidx,1} = reshape( predInstanceMaskMat_loops{loopidx,1}, [feaMapSize(1), feaMapSize(2), 3] );
        predInstanceMaskMat_loops{loopidx,1} = rescaleFeaMap(predInstanceMaskMat_loops{1});
        predInstanceMaskMat_loops{loopidx,1} = imresize(predInstanceMaskMat_loops{loopidx,1}, [sz(1),sz(2)]);
        
        rng(77); randProj = randn(feaMapSize(3), 3);
        predInstanceMaskMat_loops{loopidx,2} = reshape(feaMap, [], size(feaMap,3));
        predInstanceMaskMat_loops{loopidx,2} = predInstanceMaskMat_loops{loopidx,2}*randProj;
        predInstanceMaskMat_loops{loopidx,2} = reshape( predInstanceMaskMat_loops{loopidx,2}, [feaMapSize(1), feaMapSize(2), 3] );
        predInstanceMaskMat_loops{loopidx,2} = rescaleFeaMap(predInstanceMaskMat_loops{loopidx,2});
        predInstanceMaskMat_loops{loopidx,2} = imresize(predInstanceMaskMat_loops{loopidx,2}, [sz(1),sz(2)]);
        
        rng(7); randProj = randn(feaMapSize(3), 3);
        predInstanceMaskMat_loops{loopidx,3} = reshape(feaMap, [], size(feaMap,3));
        predInstanceMaskMat_loops{loopidx,3} = predInstanceMaskMat_loops{loopidx,3}*randProj;
        predInstanceMaskMat_loops{loopidx,3} = reshape( predInstanceMaskMat_loops{loopidx,3}, [feaMapSize(1), feaMapSize(2), 3] );
        predInstanceMaskMat_loops{loopidx,3} = rescaleFeaMap(predInstanceMaskMat_loops{loopidx,3});
        predInstanceMaskMat_loops{loopidx,3} = imresize(predInstanceMaskMat_loops{loopidx,3}, [sz(1),sz(2)]);
    end
    
    %% show figures
    imgFig = figure(1);
    set(imgFig, 'Position', [100 100 1500 800]) % [1 1 width height]
    subplot(2,3,1);
    imshow(uint8(img)); title('original image');
    subplot(2,3,2);
    imagesc(gtOrg); axis off image; title('instance mask');
    subplot(2,3,4);
    imagesc(predInstanceMaskMat_loop0{1}); axis off image; title('Loop-0 randProj 3-dim')
    subplot(2,3,5);
    imagesc(predInstanceMaskMat_loop0{2}); axis off image; title('Loop-0 randProj 3-dim')
    subplot(2,3,6);
    imagesc(predInstanceMaskMat_loop0{3}); axis off image; title('Loop-0 randProj 3-dim')
    export_fig(sprintf('%s/id%d_summary.jpg', prefix4saving, idx));
    
    imgFig3 = figure(2);
    set(imgFig3, 'Position', [100 100 1000 1000]) % [1 1 width height]
    winH = 4;
    winW = 3;
    curWinIdx = 1;
    for loopidx = 1:5:loopNum
        subplot(winH,winW,curWinIdx); curWinIdx = curWinIdx + 1;
        imagesc(predInstanceMaskMat_loops{loopidx,1}); axis off image; title(sprintf('Loop-%d randProj (3-dim)',loopidx))
        subplot(winH,winW,curWinIdx); curWinIdx = curWinIdx + 1;
        imagesc(predInstanceMaskMat_loops{loopidx,2}); axis off image; title(sprintf('Loop-%d randProj (3-dim)',loopidx))
        subplot(winH,winW,curWinIdx); curWinIdx = curWinIdx + 1;
        imagesc(predInstanceMaskMat_loops{loopidx,3}); axis off image; title(sprintf('Loop-%d randProj (3-dim)',loopidx))
    end
    export_fig(sprintf('%s/id%d_summaryLoops.jpg', prefix4saving, idx));     
end
%% leaving blank
