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

addpath(genpath('../libs'))
addpath('./local_functions_demo1')

path_to_matconvnet = '../libs/matconvnet-1.0-beta23_modifiedDagnn';
run(fullfile(path_to_matconvnet, 'matlab', 'vl_setupnn'));
addpath(genpath(fullfile('dependencies', 'matconvnet','examples')));

dataDir = './dataset';
%% read matconvnet model
% set GPU
flagSaveFig = true; % {true false} whether to store the result

h = 28;
w = 28;
panelSZ = round(64/2)*2;
stepSize = (panelSZ-h)/2;
posList = [1, stepSize, 2*stepSize];
posList = repmat(posList, 3, 1);
posList = cat(3, posList, posList');
posList = reshape(posList, [], 2);
N = size(posList,1)+1;
%% prepare imdb
imdb.path_to_dataset = './toydata_v3';

fileList = dir( fullfile(imdb.path_to_dataset, 'batch*mat') );
imdb.imgList = {fileList.name};
imdb.set = ones(1, length(imdb.imgList));


fileList = dir( fullfile(imdb.path_to_dataset, 'test*mat') );
imdb.imgList(end+1:end+length({fileList.name})) = {fileList.name};
imdb.set(end+1:end+length({fileList.name})) = 2;

imdb.sets.name = {'train', 'val'};

% meanValue = 0;
% for i = 1:sum(imdb.set==1)
%     if mod(i,100) == 0
%         fprintf('\t%d/%d\n', i, sum(imdb.set==1));
%     end
%     tmpMat = load(fullfile(imdb.path_to_dataset, imdb.imgList{i}));
%     meanValue = meanValue + mean(tmpMat.imgMat(:));
% end
% meanValue = meanValue / sum(imdb.set==1);

meanValue = reshape([0.0717, 0.0820, 0.0715], [1 1 3]);
imdb.meta.meanvalue = meanValue; % 0.0981
imdb.meta.className = {};
for i = 1:9
    imdb.meta.className{end+1} = int2str(i);
end
imdb.meta.className{end+1} = '10';
imdb.meta.className{end+1} = 'void';
imdb.meta.classNum = length(imdb.meta.className);
imdb.meta.height = panelSZ;
imdb.meta.width = panelSZ;

save('imdb_toydata_v3_from_mnist.mat', 'imdb')




