classdef InstanceSegMMLoss_randSample < dagnn.Loss      
    properties        
        marginAlpha_
        adaptiveMM = false;
    end
    properties (Transient)
        curSimMat
        weightMat
        marginMat
        mass_
        size_
	lastLayerName
    end
    
    methods
        function outputs = forward(obj, inputs, params)            
            [h, w, ch, bs] = size(inputs{1});
            sz = [h, w, ch, bs];
            mass = sz(1) * sz(2) + 1;
            
            obj.size_ = sz;
            obj.mass_ = mass;
            
            layerName = obj.lastLayerName; %'res7_cosSim';
            if isnan(obj.net.getLayerIndex(layerName))
                layerName = 'obj_instSeg_reg';
            end
            obj.curSimMat = obj.net.layers(obj.net.getLayerIndex(layerName)).block.curSimMat;
            obj.weightMat = obj.net.layers(obj.net.getLayerIndex(layerName)).block.weightMat;
            
            
            if obj.adaptiveMM
                obj.marginMat = obj.net.layers(obj.net.getLayerIndex(layerName)).block.marginMat;
                obj.marginMat = max(obj.marginMat*obj.marginAlpha_, obj.marginAlpha_*0.2);
                outputs{1} = vl_nnloss_modified(inputs{1}, obj.curSimMat, 'marginAlpha_', obj.marginAlpha_, ...
                'loss', obj.loss, ...
                'instanceWeights', obj.weightMat, 'marginMat', obj.marginMat ) ; % 1./mass
            else                
                outputs{1} = vl_nnloss_modified(inputs{1}, obj.curSimMat, 'marginAlpha_', obj.marginAlpha_, ...
                    'loss', obj.loss, ...
                    'instanceWeights', obj.weightMat ) ; % 1./mass
            end
            n = obj.numAveraged ;
            m = n + size(inputs{1},4) ;
            obj.average = (n * obj.average + double(gather(outputs{1}))) / m ;
            obj.numAveraged = m ;
        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            derInputs{1} = vl_nnloss_modified(inputs{1}, obj.curSimMat, derOutputs{1}, 'marginAlpha_', obj.marginAlpha_, ...
                'loss', obj.loss, ...
                'instanceWeights', obj.weightMat ) ; % 1./mass
            derInputs{2} = [] ;
            derParams = {} ;
        end
        
        function obj = InstanceSegMMLoss_randSample(varargin)
            obj.load(varargin) ;
        end
    end
end
