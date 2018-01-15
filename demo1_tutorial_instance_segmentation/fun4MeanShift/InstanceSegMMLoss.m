classdef InstanceSegMMLoss < dagnn.Loss      
    properties        
        marginAlpha_
    end
    properties (Transient)
        curSimMat
        weightMat
        mass_
        size_
    end
    
    methods
        function outputs = forward(obj, inputs, params)            
            [h, w, ch, bs] = size(inputs{1});
            sz = [h, w, ch, bs];
            mass = sz(1) * sz(2) + 1;
            
            obj.size_ = sz;
            obj.mass_ = mass;
            
%             gpuMode = isa(inputs{1}, 'gpuArray');
%             if gpuMode
%                 grndLabel = gpuArray(zeros(sz, 'single'));
%             else
%                 grndLabel = zeros(sz, 'single');
%             end
%             
%             for j = 1:sz(4)
%                 C = inputs{2}(:,:,:,j);
%                 C = reshape(C, [sz(1), 1]);                
%                 C = repmat(C, 1, sz(2));
%                 C = (C==C');
%                 grndLabel(:,:,:,j) = C;
%             end            
%             obj.curSimMat = grndLabel;
            
            layerName = 'obj_instSeg';
            if isnan(obj.net.getLayerIndex(layerName))
                layerName = 'obj_instSeg_reg';
            end
            obj.curSimMat = obj.net.layers(obj.net.getLayerIndex(layerName)).block.curSimMat;
            obj.weightMat = obj.net.layers(obj.net.getLayerIndex(layerName)).block.weightMat;
            
            outputs{1} = vl_nnloss(inputs{1}, obj.curSimMat, 'marginAlpha_', obj.marginAlpha_, ...
                'loss', obj.loss, ...
                'instanceWeights', obj.weightMat ) ; % 1./mass
            n = obj.numAveraged ;
            m = n + size(inputs{1},4) ;
            obj.average = (n * obj.average + double(gather(outputs{1}))) / m ;
            obj.numAveraged = m ;
        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
%             sz = obj.size_;
%             mass = obj.mass_; 
%             grndLabel = obj.curSimMat;
            
            derInputs{1} = vl_nnloss(inputs{1}, obj.curSimMat, derOutputs{1}, 'marginAlpha_', obj.marginAlpha_, ...
                'loss', obj.loss, ...
                'instanceWeights', obj.weightMat ) ; % 1./mass
            derInputs{2} = [] ;
            derParams = {} ;
        end
        
        function obj = InstanceSegMMLoss(varargin)
            obj.load(varargin) ;
        end
    end
end
