classdef SegmentationLossLogistic < dagnn.Loss    
    methods
        function outputs = forward(obj, inputs, params)
            sz = size(inputs{2});
            mass = sz(1) * sz(2) + 1;
            
            outputs{1} = vl_nnloss(inputs{1}, inputs{2}, [], ...
                'loss', obj.loss, ...
                'instanceWeights', 1./mass) ;
            n = obj.numAveraged ;
            m = n + size(inputs{1},4) ;
            obj.average = (n * obj.average + double(gather(outputs{1}))) / m ;
            obj.numAveraged = m ;
        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            sz = size(inputs{2});
            mass = sz(1) * sz(2) + 1;
            derInputs{1} = vl_nnloss(inputs{1}, inputs{2}, derOutputs{1}, ...
                'loss', obj.loss, ...
                'instanceWeights', 1./mass) ;
            derInputs{2} = [] ;
            derParams = {} ;
        end
        
        function obj = SegmentationLossLogistic(varargin)
            obj.load(varargin) ;
        end
    end
end
