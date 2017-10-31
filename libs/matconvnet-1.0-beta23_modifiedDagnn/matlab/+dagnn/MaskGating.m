classdef MaskGating < dagnn.ElementWise
    properties
        size
        hasBias = false        
    end
    properties (Transient)
        numInputs
    end
    
    methods
        
        function outputs = forward(obj, inputs, params)
            params = inputs{end};
            inputs = inputs(1:end-1);
            obj.numInputs = numel(inputs) ;
            
%             figure(1);
%             for i = 1:5
%                 subplot(2,3,i+1); imagesc(params(:,:,i)); axis image off;
%             end
%             figure(2); channelIdx = 10;
%             for i = 1:5
%                 subplot(2,3,i); imagesc(inputs{i}(:,:,channelIdx)); axis image off;
%             end
            
            
            outputs{1} = 0;
            for k = 1:numel(inputs)
                outputs{1} = outputs{1} + bsxfun(@times, inputs{k}, params(:,:,k)) ;
            end                    
        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            derInputs = cell(1, numel(inputs));
            params = inputs{end};
            inputs = inputs(1:end-1);
            derInputs{end} = gpuArray(zeros(size(derOutputs{1},1), size(derOutputs{1},2), obj.numInputs, 'single'));
            for k = 1:obj.numInputs
                derInputs{k} = bsxfun(@times, derOutputs{1}, params(:,:,k) );
                A = bsxfun(@times, inputs{k}, derOutputs{1});
                derInputs{end}(:,:,k) = sum(A,3);
            end
            derParams = {} ;
            %derInputs{end+1} = [];
        end
        
        function obj = Scale(varargin)
            obj.load(varargin) ;
        end
    end
end
