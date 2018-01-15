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
            
            outputs{1} = 0;
            for k = 1:numel(inputs)
                outputs{1} = outputs{1} + bsxfun(@times, inputs{k}, params(:,:,k)) ;
            end                    
        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            params = inputs{end};
            inputs = inputs(1:end-1);
            
            for k = 1:obj.numInputs
                derInputs{k} = bsxfun(@times, derOutputs{1}, params(:,:,k) );
            end
            derParams = {} ;
            derInputs{end+1} = [];
        end
        
        function obj = Scale(varargin)
            obj.load(varargin) ;
        end
    end
end
