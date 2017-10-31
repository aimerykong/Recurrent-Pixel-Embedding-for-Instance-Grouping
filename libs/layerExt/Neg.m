classdef Neg < dagnn.ElementWise
    methods
        function outputs = forward(obj, inputs, params)
            outputs{1} = -inputs{1};
        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            derInputs{1} = -1 * derOutputs{1};
            derParams = {} ;
        end
        
    end
end
