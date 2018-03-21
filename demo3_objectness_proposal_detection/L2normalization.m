classdef L2normalization < dagnn.ElementWise
    properties (Transient)
        numInputs
        SIZE_
    end
    
    methods        
        function outputs = forward(obj, inputs, params)
            obj.numInputs = numel(inputs);
            assert(obj.numInputs == 1);
            
            X = inputs{1};
            obj.SIZE_ = size(X);
            if length(obj.SIZE_) < 3
                obj.SIZE_ = [obj.SIZE_, 1, 1];
            elseif length(obj.SIZE_) ==3
                obj.SIZE_ = [obj.SIZE_ 1];
            elseif length(obj.SIZE_) > 4
                error('length(SIZE_) > 4');
            end
            
            X = permute(X, [3,1,2,4]);            
            X = reshape(X, [size(X,1), prod(obj.SIZE_)/obj.SIZE_(3)] );            
            sumX = sqrt(sum(X.^2,1));
            %sumX = sumX + (sumX==0);
            sumX = sumX + 0.00001;
            X = bsxfun(@rdivide, X, sumX);
                        
            X = reshape(X, [size(X,1), obj.SIZE_(1:2), obj.SIZE_(4)]);
            X = permute(X, [2,3,1,4]);        
            outputs{1} = X;
        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            derInputs = cell(1, numel(inputs));
                        
            dzdy = derOutputs{1};            
            X = inputs{1};
%             obj.SIZE_ = size(X);
%             if length(obj.SIZE_) < 3
%                 obj.SIZE_ = [obj.SIZE_, 1, 1];
%             elseif length(obj.SIZE_) ==3
%                 obj.SIZE_ = [obj.SIZE_ 1];
%             elseif length(obj.SIZE_) > 4
%                 error('length(SIZE_) > 4');
%             end
            
            X = permute(X, [3,1,2,4]);
            X = reshape(X, [size(X,1), prod(obj.SIZE_)/obj.SIZE_(3)] );
            dzdy = permute(dzdy, [3,1,2,4]);            
            dzdy = reshape(dzdy, [size(dzdy,1), prod(obj.SIZE_)/obj.SIZE_(3)] );
                        
            lambda = 1./(sqrt(sum(X.^2, 1)) + 1e-10);            
            dzdx = bsxfun(@times, lambda, dzdy) - bsxfun(@times, X, (lambda.^3) .* sum(X.*dzdy, 1));            
            
            dzdx = reshape(dzdx, [size(dzdx,1), obj.SIZE_(1:2), obj.SIZE_(4)]);
            dzdx = permute(dzdx, [2,3,1,4]);  
            
            derInputs{1} = dzdx;                      
            derParams = {} ;            
        end
        
        function obj = Scale(varargin)
            obj.load(varargin) ;
        end
    end
end
