clear;
clc;
close all;

addpath './fun4MeanShift';
addpath('./local_functions_demo1')
addpath '../libs/exportFig/';

path_to_matconvnet = '../libs/matconvnet-1.0-beta23_modifiedDagnn';
run(fullfile(path_to_matconvnet, 'matlab', 'vl_setupnn'));
addpath(genpath(fullfile('dependencies', 'matconvnet','examples')));

dataDir = './dataset';
%% download mnist
files = {'train-images-idx3-ubyte', ... 
    'train-labels-idx1-ubyte', ...
    't10k-images-idx3-ubyte', ...
    't10k-labels-idx1-ubyte'} ;

if ~exist(dataDir, 'dir')
    mkdir(dataDir) ;    
end
for i=1:4
    if ~exist(fullfile(dataDir, files{i}), 'file')
        url = sprintf('http://yann.lecun.com/exdb/mnist/%s.gz',files{i}) ;
        fprintf('downloading %s\n', url) ;
        gunzip(url, dataDir) ;
    end
end
%% generate data
generate_train_set_part1();
generate_train_set_part2();
generate_test_set();
%% visualize the generated data
if ~exist('trainImages', 'var')
    trainImages = loadMNISTImages(fullfile(dataDir, 'train-images-idx3-ubyte'));
    trainLabels = loadMNISTLabels(fullfile(dataDir, 'train-labels-idx1-ubyte'));
    testImages = loadMNISTImages(fullfile(dataDir, 't10k-images-idx3-ubyte'));
    testLabels = loadMNISTLabels(fullfile(dataDir, 't10k-labels-idx1-ubyte'));
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


h = 28;
w = 28;
panelSZ = round(64/2)*2;
stepSize = (panelSZ-h)/2;
posList = [1, stepSize, 2*stepSize];
posList = repmat(posList, 3, 1);
posList = cat(3, posList, posList');
posList = reshape(posList, [], 2);
N = size(posList,1)+1;
%% validate a sample
rng(777);

panelVoid4img = zeros(panelSZ, panelSZ, 3, 'single');
panelVoid4semanticMask = zeros(panelSZ, panelSZ, 1, 'single');
panelVoid4instanceMask = zeros(panelSZ, panelSZ, 1, 'single');

curPosList = randperm(size(posList,1));
curPosList = posList(curPosList,:);
curN = randperm(N,1)-1;

% curPosList = posList;
% curN = 9;
for i = 1:curN
    curIdx = randperm(size(trainImages,3), 1);
    curImg = trainImages(:,:,curIdx);
    curSemnMask = trainSemanticMask(:,:,curIdx);
    curPos = curPosList(i,:); % [y, x]
    %% image
    curImg_R = curImg*rand(1);
    curImg_B = curImg*rand(1);
    curImg_G = curImg*rand(1);
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
    tmpMat(a) = i;
    panelVoid4instanceMask(curPos(1):curPos(1)+h-1, curPos(2):curPos(2)+w-1, :) = tmpMat;
end
%% visualization
figure(1);
subplot(1,3,1);
imagesc(panelVoid4img); axis off image; colorbar; title('digit with random color');
subplot(1,3,2);
imagesc(panelVoid4semanticMask); axis off image; caxis([0 11]); colorbar; title('semantic mask');
subplot(1,3,3);
imagesc(panelVoid4instanceMask); axis off image; caxis([0 10]); colorbar; title('instance mask');

fprintf('done!');