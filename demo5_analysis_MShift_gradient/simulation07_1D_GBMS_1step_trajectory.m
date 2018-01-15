clear
clc;
close all;
rng(777);
%% parameter config
% the classifier: y = w*x+b 
w=0.5;
b=-0.5;

% for mean shift
delta = 0.2;
MAXITERATION = 5;
meanShiftNumber = 2; % [1~7]
stepSize = 0.1; % learning rate
MAXITERATION_tweakingX = 30; % #iterations to update x
%% generate 3 mixed Gaussians with different mean and variance
mu1 = 3; var1 = 0.2;
mu2 = 4; var2 = 0.3;
mu3 = 5; var3 = 0.1;

x1 = mu1 + var1.*randn(1, 1000);
y1 = ones(1, 1000);
x2 = mu2 + var2.*randn(1, 1000);
y2 = 2*ones(1, 1000);
x3 = mu3 + var3.*randn(1, 1000);
y3 = 2*ones(1, 1000);
x = [x1, x2, x3];
y = [y1, y2, y3];
y = (x>=3.5)+1;
%% add tools
path_to_matconvnet = '../libs/matconvnet-1.0-beta23_modifiedDagnn/';
run(fullfile(path_to_matconvnet, 'matlab', 'vl_setupnn'));
addpath(genpath(fullfile(path_to_matconvnet,'examples')));
addpath('./fun4MShift_analysis');
addpath('../libs/exportFig');

net = dagnn.DagNN();

sName = 'data';
lName = 'dummyConv';
block = dagnn.Conv('size', [1 1 1 1], 'hasBias', false, 'stride', 1, 'pad', 0, 'dilate', 1);
net.addLayer(lName, block, sName, lName, {[lName '_f']});
ind = net.getParamIndex([lName '_f']);
net.params(ind).value = single(1);
net.params(ind).weightDecay = 0;
net.params(ind).learningRate = 0;
sName = lName;


lName = 'dummyRelu';
block = dagnn.ReLU('leak', 0);
net.addLayer(lName, block, sName, lName);
sName = lName;


lossList = {};
baseSName = sName;

lName = 'regressLayer';
block = dagnn.Conv('size', [1 1 1 1], 'hasBias', true, 'stride', 1, 'pad', 0, 'dilate', 1);
net.addLayer(lName, block, sName, lName, {[lName '_f'], [lName '_b']});
ind = net.getParamIndex([lName '_f']);
net.params(ind).value = single(0.5);
net.params(ind).weightDecay = 0;
net.params(ind).learningRate = 0;
ind = net.getParamIndex([lName '_b']);
net.params(ind).value = single(-0.5);
net.params(ind).weightDecay = 0;
net.params(ind).learningRate = 0;
sName = lName;

for loopIdx = 1:meanShiftNumber
    GaussianBandwidth = 0.1;
    [net, sName] = addOneLoop_forMeanShiftGrouping(net, baseSName, loopIdx, GaussianBandwidth);
    baseSName = sName;
    
    lName = sprintf('regressLayer_loop%d', loopIdx); 
    block = dagnn.Conv('size', [1 1 1 1], 'hasBias', true, 'stride', 1, 'pad', 0, 'dilate', 1);
    net.addLayer(lName, block, sName, lName, {[lName '_f'], [lName '_b']});
    ind = net.getParamIndex([lName '_f']);
    net.params(ind).value = single(0.5);
    net.params(ind).weightDecay = 0;
    net.params(ind).learningRate = 0;
    ind = net.getParamIndex([lName '_b']);
    net.params(ind).value = single(-0.5);
    net.params(ind).weightDecay = 0;
    net.params(ind).learningRate = 0;
    sName = lName;
    
    obj_name = sprintf('regLoss_loop%d', loopIdx);
    gt_name = 'classLabel';
    net.addLayer(obj_name, ...
        L2RegressionLoss('loss', 'regressionloss'), ... softmaxlog logistic
        {sName, gt_name}, obj_name)
    lossList{end+1} = obj_name;
    lossList{end+1} = 1;
end

net.move('gpu');
net.mode = 'train' ;

Xinput = single(reshape(x,[50,60,1,1]));
Yinput = single(reshape(y,[50,60,1,1]));
inputs = {'data', gpuArray(Xinput), 'classLabel', gpuArray(Yinput) };
net.conserveMemory = 0;
%net.eval(inputs) ;
net.eval(inputs, lossList, 'backPropAboveLayerName', 'dummyRelu') ;
%% show original data
imgFig = figure(1);
set( imgFig, 'Position', [100 100 1500 700]);

subplot(2,3,1);
x_class1 = x(x<=3.5);
x_class2 = x(x>3.5);

histogram(x_class1, 100, 'FaceColor', 'r', 'EdgeColor', 'r');
hold on;
histogram(x_class2, 100, 'FaceColor', 'b', 'EdgeColor', 'b');
title('original data with class labe');
xlim([2, 6]);
%% linear regression for classification
% y = w*x+b, where 
% w=0.5;
% b=-0.5;
% red is class-1, blue is class-2

subplot(2,3,2);
x_class1 = x(x<=3.5);
x_class2 = x(x>3.5);
plot(2:0.1:6, w*[2:0.1:6]+b, 'k-');
hold on;
plot([4 4], [0 2.5]);
title('linear classifier through regression (fixed/defined)')
xlim([2, 6]);
%% GBMS
prex = x;
X0 = prex;
curX = prex;

for curIter = 1:MAXITERATION
    fprintf('%d/%d...\n', curIter, MAXITERATION);
    tic
    %% main loop
    curX = prex;       
    %% GBMS forward    
    S = repmat(curX',1,numel(curX));
    S = S - S';    
    S = S.^2;    
    G = exp(-0.5*S/(delta^2));    
    d = sum(G,1)+0.00001;
    q = (d).^(-1);    
    P = G*diag(q);
    
    curX = curX*P;
    %% update    
    prex = curX;
    toc;
end
subplot(2,3,3);
histogram(curX, 100)
xlim([2, 6]);
title(sprintf('after %d loops of std MShift (unsupervised)', MAXITERATION));
%% supervised tweaking through MShift
gradx = gather(net.vars(2).der);
gradx = gradx(:)';
prex = x;
curX = x;
y1 = y-b;
updatedX_training_with_meanshift = {single(x)};
for curIter = 1:MAXITERATION_tweakingX
    fprintf('%d/%d...\n', curIter, 30);    
    %% main loop    
    curX = prex;
    Xinput = single(reshape(curX,[50,60,1,1]));
    Yinput = single(reshape(y,[50,60,1,1]));
    inputs = {'data', gpuArray(Xinput), 'classLabel', gpuArray(Yinput) };
    net.conserveMemory = 0;
    %net.eval(inputs) ;
    net.eval(inputs, lossList, 'backPropAboveLayerName', 'dummyRelu') ;
    
    gradx = gather(net.vars(2).der);
    gradx = gradx(:)';
    
    gradx_loss0 = 2*w*w*prex - 2*w*y1;    
    curX = prex - stepSize*gradx*numel(gradx); % *numel(gradx)
    prex = curX;    
    updatedX_training_with_meanshift{end+1} = curX;
end
x_tweaked = curX;
subplot(2,3,4);
histogram(x_tweaked, 100)
xlim([2, 6]);
%ylim([0, 200]);
title('tweaked data with GBMS (supervised)');
%% GBMS on tweaked data
prex = x_tweaked;
X0 = prex;
curX = prex;
for curIter = 1:MAXITERATION
    fprintf('%d/%d...\n', curIter, MAXITERATION);
    %% GBMS forward
    S = repmat(curX',1,numel(curX));
    S = S - S';
    S = S.^2;
    G = exp(-0.5*S./(delta^2));    
    d = sum(G, 2)+0.000001;
    q = (d).^(-1);    
    P = G*diag(q);
    
    curX = curX*P;
    %% update    
    prex = curX;
end
figure(1);
subplot(2,3,5);
histogram(curX, 100)
xlim([2, 6]);
title(sprintf('after %d loops of MShift over tweaked data', MAXITERATION));
%% linear regression for classification
% y = w*x+b, where w=0.5, b=-0.5
% red is class-1, blue is class-2
% |y-w*x-b|^2
y1 = y-b;
curX = x;
prex = x;
for curIter = 1:30
    fprintf('%d/%d...\n', curIter, 30);    
    %% main loop
    
    % | y1 - w*x |^2
    % loss wrt x = tr(w*x*x'*w') - 2*y1*x'*w'
    %            = w*w*tr(x*x') - 2*y1*x'*w'
    gradx = 2*w*w*curX - 2*w*y1;
    curX = prex - stepSize*gradx;
    prex = curX;    
end
subplot(2,3,6);
histogram(curX, 100)
xlim([2, 6]);
title('tweaked data without MShift (supervised)');

export_fig(sprintf('simulation07_GBMS_%dLoops_summary.png', meanShiftNumber));
%% draw trajectories on random points (training with mean shift)
figure(11);
rng(777);

pointNum = 100;
trajectoryMat = cell2mat(updatedX_training_with_meanshift');
randIdxList = randperm(size(trajectoryMat,2), pointNum);
trajectoryMat = trajectoryMat(:, randIdxList);

lineStyles = linspecer(pointNum);

hold on;
for i = 1:pointNum
    x = trajectoryMat(:,i)';
    y = 1:size(trajectoryMat,1);    
    plot(x,y, 'color', lineStyles(i,:));
end
xlim([2, 6]);
ylim([0, size(trajectoryMat,1)]);
title('trajectories of random points (training with meanshift)');

export_fig(sprintf('simulation07_GBMS_%dLoops.png', meanShiftNumber));
%% leaving blank for comments
%{
%}
