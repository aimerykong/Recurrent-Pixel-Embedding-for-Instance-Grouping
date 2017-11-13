function generate_train_set_part1()

%LD_LIBRARY_PATH=/usr/local/cuda-7.5/lib64:local matlab -nodisplay
%{
run(fullfile(path_to_matconvnet, 'matlab', 'vl_setupnn'));
addpath(fullfile(path_to_matconvnet, 'matlab'));
vl_compilenn('enableGpu', true, ...
               'cudaRoot', '/usr/local/cuda-7.5', ...
               'cudaMethod', 'nvcc', ...
               'enableCudnn', true, ...
               'cudnnRoot', '/usr/local/cuda-7.5/cudnn-v5') ;

%}
% clear
close all
clc;
%% load raw data
if ~exist('trainImages', 'var')
    trainImages = loadMNISTImages('dataset/train-images-idx3-ubyte');
    trainLabels = loadMNISTLabels('dataset/train-labels-idx1-ubyte');
    testImages = loadMNISTImages('dataset/t10k-images-idx3-ubyte');
    testLabels = loadMNISTLabels('dataset/t10k-labels-idx1-ubyte');
end
trainImages = single(trainImages);
testImages = single(testImages);
trainLabels = single(trainLabels);
testLabels = single(testLabels);

trainImages = reshape(trainImages, [28 28 numel(trainImages)/784]);
testImages = reshape(testImages, [28 28 numel(testImages)/784]);
testLabels(testLabels==0) = 10;
trainLabels(trainLabels==0) = 10;

trainSemanticMask = bsxfun(@times, single(trainImages>0), reshape(trainLabels, [1 1 size(trainImages,3)]) );
testSemanticMask = bsxfun(@times, single(testImages>0), reshape(testLabels, [1 1 size(testImages,3)]) );
% trainSemanticMask(trainSemanticMask==0) = 11;
% testSemanticMask(testSemanticMask==0) = 11;

rng(777);

h = 28;
w = 28;
panelSZ = round(64/2)*2;
stepSize = (panelSZ-h)/2;
posList = [1, stepSize, 2*stepSize];
posList = repmat(posList, 3, 1);
posList = cat(3, posList, posList');
posList = reshape(posList, [], 2);
N = size(posList,1)+1;
%% stitch for instance segmentation
rng(777);
colorList = rand(3, 10);
colorList = bsxfun(@rdivide, colorList, max(colorList,[],1));
colorList = colorList';

imdb.path = './toydata_v3';
imdb.imgList = {};

if ~isdir(imdb.path)
    mkdir(imdb.path);
end

desiredNumber = 500000;
batchSize = 500;
imgMat = zeros(panelSZ, panelSZ, 3, batchSize, 'single');
semanticMaskMat = zeros(panelSZ, panelSZ, batchSize, 'single');
instanceMaskMat = zeros(panelSZ, panelSZ, batchSize, 'single');

num = 25;
M = 5;
N = 5;

batchCount = 1;
j = 1;
for i = 1:desiredNumber
    if mod(i, batchSize) == 0
        fprintf('\t%d/%d\n', i, desiredNumber);
        %% visualize
        %{
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
        
        showcaseSemanticMask= semanticMaskMat(:,:,1:num);
        showcaseSemanticMask = reshape(showcaseSemanticMask, [numel(showcaseSemanticMask(:,:,1)), num]);
        showcaseSemanticMask = showStitch(showcaseSemanticMask, [panelSZ panelSZ], M, N, 'whitelines', 'linewidth', 5);
        
        showcaseInstanceMask= instanceMaskMat(:,:,1:num);
        showcaseInstanceMask = reshape(showcaseInstanceMask, [numel(showcaseInstanceMask(:,:,1)), num]);
        showcaseInstanceMask = showStitch(showcaseInstanceMask, [panelSZ panelSZ], M, N, 'whitelines', 'linewidth', 5);
        
        imgFig2 = figure(2);
        subWindowH = 1;
        subWindowW = 3;
        set(imgFig2, 'Position', [100 100 1600 700]) % [1 1 width height]
        windowID = 1;
        subplot(subWindowH, subWindowW, windowID); imagesc(showcaseSamples); axis off image; title('random samples'); colorbar; windowID = windowID + 1;
        subplot(subWindowH, subWindowW, windowID); imagesc(showcaseSemanticMask); axis off image; title('showcase SemanticMask'); caxis([0, 11]); colorbar; windowID = windowID + 1;
        subplot(subWindowH, subWindowW, windowID); imagesc(showcaseInstanceMask); axis off image; title('showcase instanceMask'); caxis([0, 4]); colorbar; windowID = windowID + 1;
        %}
        %% save
        save(fullfile(imdb.path, sprintf('batch%05d.mat', batchCount)), 'imgMat', 'semanticMaskMat', 'instanceMaskMat');
        
        imgMat = zeros(panelSZ, panelSZ, 3,  batchSize, 'single');
        semanticMaskMat = zeros(panelSZ, panelSZ, batchSize, 'single');
        instanceMaskMat = zeros(panelSZ, panelSZ, batchSize, 'single');
        
        batchCount = batchCount + 1;
        j = 1;
    end
    
    panelVoid4img = zeros(panelSZ, panelSZ, 3, 'single');
    panelVoid4semanticMask = zeros(panelSZ, panelSZ, 1, 'single');
    panelVoid4instanceMask = zeros(panelSZ, panelSZ, 1, 'single');
    
    curPosList = randperm(size(posList,1));
    curPosList = posList(curPosList,:);
    curN = randperm(N,1)-1;
    
    for iii = 1:curN
        curIdx = randperm(size(trainImages,3), 1);
        curImg = trainImages(:,:,curIdx);
        curLabel = trainLabels(curIdx);
        curSemnMask = trainSemanticMask(:,:,curIdx);
        curPos = curPosList(iii,:); % [y, x]
        %% image
        %curImg_R = curImg*max(rand(1)/10*curLabel*0.2+0.2, 0.1);
        %curImg_B = curImg*max(rand(1)/10*curLabel*0.2+0.3, 0.1);
        %curImg_G = curImg*max(rand(1)/10*(11-curLabel)*0.2+0.8, 0.1);
         
        curImg_R = curImg*colorList(curLabel,1);
        curImg_B = curImg*colorList(curLabel,2);
        curImg_G = curImg*colorList(curLabel,3);
        curImg = cat(3, curImg_R, curImg_B, curImg_G);
        
        tmpMat = panelVoid4img(curPos(1):curPos(1)+h-1, curPos(2):curPos(2)+w-1, :);
        a = find(curImg~=0);
        tmpMat(a) = curImg(a);
        
        panelVoid4img(curPos(1):curPos(1)+h-1, curPos(2):curPos(2)+w-1, :) = tmpMat;
        %% semantic mask
        tmpMat = panelVoid4semanticMask(curPos(1):curPos(1)+h-1, curPos(2):curPos(2)+w-1, :);
        a = find(curSemnMask~=0);
        tmpMat(a) = curSemnMask(a);
        panelVoid4semanticMask(curPos(1):curPos(1)+h-1, curPos(2):curPos(2)+w-1, :) = tmpMat;
        %% instance mask
        tmpMat = panelVoid4instanceMask(curPos(1):curPos(1)+h-1, curPos(2):curPos(2)+w-1, :);
        tmpMat(a) = iii;
        panelVoid4instanceMask(curPos(1):curPos(1)+h-1, curPos(2):curPos(2)+w-1, :) = tmpMat;
    end
    %% save
    imgMat(:,:,:,j) = panelVoid4img;
    semanticMaskMat(:,:,j) = panelVoid4semanticMask;
    instanceMaskMat(:,:,j) = panelVoid4instanceMask;
    j = j + 1;
end
%% visualize
%{
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

showcaseSemanticMask= semanticMaskMat(:,:,1:num);
showcaseSemanticMask = reshape(showcaseSemanticMask, [numel(showcaseSemanticMask(:,:,1)), num]);
showcaseSemanticMask = showStitch(showcaseSemanticMask, [panelSZ panelSZ], M, N, 'whitelines', 'linewidth', 5);

showcaseInstanceMask= instanceMaskMat(:,:,1:num);
showcaseInstanceMask = reshape(showcaseInstanceMask, [numel(showcaseInstanceMask(:,:,1)), num]);
showcaseInstanceMask = showStitch(showcaseInstanceMask, [panelSZ panelSZ], M, N, 'whitelines', 'linewidth', 5);

imgFig2 = figure(2);
subWindowH = 1;
subWindowW = 3;
set(imgFig2, 'Position', [100 100 1600 700]) % [1 1 width height]
windowID = 1;
subplot(subWindowH, subWindowW, windowID); imagesc(showcaseSamples); axis off image; title('random samples'); colorbar; windowID = windowID + 1;
subplot(subWindowH, subWindowW, windowID); imagesc(showcaseSemanticMask); axis off image; title('showcase SemanticMask'); caxis([0, 11]); colorbar; windowID = windowID + 1;
subplot(subWindowH, subWindowW, windowID); imagesc(showcaseInstanceMask); axis off image; title('showcase instanceMask'); caxis([0, 4]); colorbar; windowID = windowID + 1;
%}
%% leaving blank









