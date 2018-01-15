classdef InstanceSegRegLoss < dagnn.Loss    
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
            
            gpuMode = isa(inputs{1}, 'gpuArray');
            if gpuMode
                obj.curSimMat = gpuArray(zeros(sz, 'single'));
                obj.weightMat = gpuArray(zeros(sz, 'single'));
            else
                obj.curSimMat = zeros(sz, 'single');
                obj.weightMat = zeros(sz, 'single');
            end
            
            for j = 1:sz(4)
                C = inputs{2}(:,:,:,j);
                W = C;
                for ii = 0:4
                    W(W==ii) = sum(C(:)==ii);
                end
                W = W(:)*W(:)';
                W = 1 ./ W;
                W = W ./ sum(W(:));
                
                C = reshape(C, [sz(1), 1]);                
                C = repmat(C, 1, sz(2));
                C = (C==C');
                obj.curSimMat(:,:,:,j) = C;
                obj.weightMat(:,:,:,j) = W;   
            end            
            
            outputs{1} = vl_nnloss(inputs{1}, obj.curSimMat, [], ...
                'loss', obj.loss, ...
                'instanceWeights', obj.weightMat) ;% 1./mass
            n = obj.numAveraged ;
            m = n + size(inputs{1},4) ;
            obj.average = (n * obj.average + double(gather(outputs{1}))) / m ;
            obj.numAveraged = m ;
        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            sz = obj.size_;
            mass = obj.mass_;            
            
            gpuMode = isa(inputs{1}, 'gpuArray');
            if gpuMode
                grndLabel = gpuArray(zeros(sz, 'single'));
            else
                grndLabel = zeros(sz, 'single');
            end
            grndLabel = obj.curSimMat;
            
            derInputs{1} = vl_nnloss(inputs{1}, obj.curSimMat, derOutputs{1}, ...
                'loss', obj.loss, ...
                'instanceWeights', obj.weightMat) ;% 1./mass
            derInputs{2} = [] ;
            derParams = {} ;
        end
        
        function obj = InstanceSegRegLoss(varargin)
            obj.load(varargin) ;
        end
    end
end
