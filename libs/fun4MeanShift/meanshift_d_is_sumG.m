classdef meanshift_d_is_sumG < dagnn.ElementWise
    properties % (Transient)
        numInputs
        SIZE_
    end
    
    % [TODO]: current version only supports batchSize=1; need to extend to 
    % multiple input images    
    methods        
        function outputs = forward(obj, inputs, params)
            obj.numInputs = numel(inputs);
%             [hw, ~, ~, bs] = size(inputs{1});
%             obj.SIZE_ = inputs{2};
%             gpuMode = isa(inputs{1}, 'gpuArray');            
%             if gpuMode
%                 Y = gpuArray(zeros(hw, 1, 1, bs, 'single'));
%             else
%                 Y = zeros(hw, 1, 1, bs, 'single');
%             end
            outputs{1} = sum(inputs{1}, 2)+0.000001;
        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            derInputs = cell(1, numel(inputs));            
            dzdy = derOutputs{1};
            gpuMode = isa(inputs{1}, 'gpuArray');            
            if gpuMode
                dzdx = gpuArray(zeros(size(inputs{1},1), size(inputs{1},2), size(inputs{1},3), size(inputs{1},4), 'single'));
            else
                dzdx = zeros(size(inputs{1},1), size(inputs{1},2), size(inputs{1},3), size(inputs{1},4), 'single');
            end
            for i = 1:size(inputs{1},4)
                dzdx(:,:,:,i) = repmat(dzdy(:,:,:,i)', size(dzdx,1), 1);
            end            
            derInputs{1} = dzdx;
            derParams = {} ;            
        end
        
        function obj = meanshift_d_is_sumG(varargin)
            obj.load(varargin) ;
        end
    end
end
