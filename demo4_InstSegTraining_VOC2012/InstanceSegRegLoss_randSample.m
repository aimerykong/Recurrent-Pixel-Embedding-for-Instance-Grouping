classdef InstanceSegRegLoss_randSample < dagnn.Loss    
    properties (Transient)
        curSimMat
        weightMat
        mass_
        size_
	lastLayerName
    end
    
    methods
        function outputs = forward(obj, inputs, params)            
	    %disp(obj)
            [h, w, ch, bs] = size(inputs{1});
            sz = [h, w, ch, bs];
            mass = sz(1) * sz(2) + 1;
            
            obj.size_ = sz;
            obj.mass_ = mass;            
            
            layerName = obj.lastLayerName; %'res7_cosSim';
            obj.curSimMat = obj.net.layers(obj.net.getLayerIndex(layerName)).block.curSimMat;
            obj.weightMat = obj.net.layers(obj.net.getLayerIndex(layerName)).block.weightMat;
                        
            outputs{1} = vl_nnloss_modified(inputs{1}, obj.curSimMat, [], ...
                'loss', obj.loss, ...
                'instanceWeights', obj.weightMat) ;% 1./mass
            n = obj.numAveraged ;
            m = n + size(inputs{1},4) ;
            obj.average = (n * obj.average + double(gather(outputs{1}))) / m ;
            obj.numAveraged = m ;
        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            derInputs{1} = vl_nnloss_modified(inputs{1}, obj.curSimMat, derOutputs{1}, ...
                'loss', obj.loss, ...
                'instanceWeights', obj.weightMat) ;% 1./mass
            derInputs{2} = [] ;
            derParams = {} ;
        end
        
        function obj = InstanceSegRegLoss_randSample(varargin)
            obj.load(varargin) ;
        end
    end
end
