classdef meanshift_q_is_inv_d < dagnn.ElementWise    
    properties (Transient)
        numInputs
        SIZE_
    end
    
    % [TODO]: current version only supports batchSize=1; need to extend to 
    % multiple input images    
    methods        
        function outputs = forward(obj, inputs, params)
            obj.numInputs = numel(inputs);
%             obj.SIZE_ = inputs{2};
            outputs{1} = (inputs{1}+0.0000001).^(-1);
        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            derInputs = cell(1, numel(inputs));
            dzdy = derOutputs{1};
            derInputs{1} = dzdy .* (-1*(inputs{1}.^(-2)));
            derParams = {} ;            
        end
        
        function obj = meanshift_q_is_inv_d(varargin)
            obj.load(varargin) ;
        end
    end
end
