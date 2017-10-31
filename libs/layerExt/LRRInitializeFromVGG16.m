function net = LRRInitializeFromVGG16(opts)
% This file is from matconvnet-FCN repository (https://github.com/vlfeat/matconvnet-fcn)
% and only has minor midifications.

opts.sourceModelUrl = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-16.mat';
opts.sourceModelPath = fullfile(opts.modelsDir, 'ini_models/imagenet-vgg-verydeep-16.mat');

% -------------------------------------------------------------------------
%                    Load & download the source model if needed (VGG VD 16)
% -------------------------------------------------------------------------
opts.sourceModelPath
if ~exist(opts.sourceModelPath, 'file')
  fprintf('%s: downloading %s\n', opts.sourceModelUrl) ;
  mkdir(fileparts(opts.sourceModelPath)) ;
  urlwrite(opts.sourceModelUrl, opts.sourceModelPath) ;
end
net = load(opts.sourceModelPath) ;

% -------------------------------------------------------------------------
%                                  Edit the model to create the FCN version
% -------------------------------------------------------------------------

% Add dropout to the fully-connected layers in the source model
drop1 = struct('name', 'dropout1', 'type', 'dropout', 'rate' , 0.5) ;
drop2 = struct('name', 'dropout2', 'type', 'dropout', 'rate' , 0.5) ;
net.layers = [net.layers(1:33) drop1 net.layers(34:35) drop2 net.layers(36:end)] ;

% Convert the model from SimpleNN to DagNN
net = dagnn.DagNN.fromSimpleNN(net, 'canonicalNames', true) ;

% Add more padding to the input layer
%net.layers(1).block.pad = 100 ;
if 1
net.layers(5).block.pad = [0 1 0 1] ;
net.layers(10).block.pad = [0 1 0 1] ;
net.layers(17).block.pad = [0 1 0 1] ;
net.layers(24).block.pad = [0 1 0 1] ;
net.layers(31).block.pad = [0 1 0 1] ;
net.layers(32).block.pad = [3 3 3 3] ;
end
% ^-- we could do [2 3 2 3] but that would not use CuDNN

% Modify the bias learning rate for all layers
for i = 1:numel(net.layers)-1
  if (isa(net.layers(i).block, 'dagnn.Conv') && net.layers(i).block.hasBias)
    filt = net.getParamIndex(net.layers(i).params{1}) ;
    bias = net.getParamIndex(net.layers(i).params{2}) ;
    net.params(bias).learningRate = 2 * net.params(filt).learningRate ;
  end
end

net.removeLayer('prob') ;
net.removeLayer('fc8') ;
