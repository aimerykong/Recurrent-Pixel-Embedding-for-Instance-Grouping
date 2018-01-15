classdef meanshift_G_is_Gaussian < dagnn.ElementWise
    properties        
        delta=0.1
    end
    properties % (Transient)
        numInputs
        SIZE_=[]
    end
    
    % [TODO]: current version only supports batchSize=1; need to extend to 
    % multiple input images    
    methods        
        function outputs = forward(obj, inputs, params)
            obj.numInputs = numel(inputs);
            % obj.SIZE_ = inputs{2};
            outputs{1} = exp((inputs{1}-1)/(obj.delta^2));
        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            derInputs = cell(1, numel(inputs));
            dzdy = derOutputs{1};
            S = inputs{1};
            derInputs{1} = dzdy.* ((1/obj.delta^2)* exp((S-1)/(obj.delta^2)));
            derParams = {} ;            
        end
        
        function obj = meanshift_G_is_Gaussian(varargin)
            obj.load(varargin) ;
        end
    end
end
