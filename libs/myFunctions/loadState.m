function [net, stats] = loadState(fileName)
% -------------------------------------------------------------------------
load(fileName, 'net', 'stats') ;
net = dagnn.DagNN.loadobj(net) ;