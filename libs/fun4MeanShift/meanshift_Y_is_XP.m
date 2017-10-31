classdef meanshift_Y_is_XP < dagnn.ElementWise
    properties (Transient)
        numInputs
        SIZE_
    end
    
    % [TODO]: current version only supports batchSize=1; need to extend to 
    % multiple input images    
    methods        
        function outputs = forward(obj, inputs, params)
            obj.numInputs = numel(inputs);
            X = inputs{1};            
            P = inputs{2};
            gpuMode = isa(X, 'gpuArray');           
            % [height width channels batchsize]
            [h, w, ch, bs] = size(X);
            obj.SIZE_ = [h, w, ch, bs]; 
            if gpuMode
                Y = gpuArray(zeros(h, w, ch, bs, 'single'));
            else
                Y = zeros(h, w, ch, bs, 'single');
            end
            for i = 1:bs
                cur_X = X(:,:,:,i);
                cur_P = P(:,:,:,i);
                cur_X = reshape(cur_X, [h*w, ch]);
                cur_X = cur_X'*cur_P;
                cur_X = reshape(cur_X', [h, w, ch]);
                Y(:,:,:,i) = cur_X;
            end
            outputs{1} = Y;
        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            derInputs = cell(1, numel(inputs));
            h = obj.SIZE_(1);
            w = obj.SIZE_(2);
            ch = obj.SIZE_(3);
            bs = obj.SIZE_(4);
            
            dzdy = derOutputs{1};            
            X = inputs{1}; 
            P = inputs{2};
            gpuMode = isa(X, 'gpuArray');
            if gpuMode
                dzdx_X = gpuArray(zeros(size(X,1), size(X,2), size(X,3), size(X,4), 'single'));
                dzdx_P = gpuArray(zeros(size(P,1), size(P,2), size(P,3), size(P,4), 'single'));
            else
                dzdx_X = zeros(size(X,1), size(X,2), size(X,3), size(X,4), 'single');
                dzdx_P = zeros(size(P,1), size(P,2), size(P,3), size(P,4), 'single');
            end
            for i = 1:size(X,4)
                cur_X = X(:,:,:,i);
                cur_P = P(:,:,:,i);
                
                cur_X = reshape(cur_X, [h*w, ch]);
                cur_X = cur_X';
                
                cur_dzdy = dzdy(:,:,:,i);
                cur_dzdy = reshape(cur_dzdy, [h*w, ch])';
                
                cur_dzdx_X = cur_dzdy*cur_P';
                cur_dzdx_P = cur_X'*cur_dzdy;
                
                cur_dzdx_X = reshape(cur_dzdx_X', [h, w, ch]);
                
                dzdx_X(:,:,:,i) = cur_dzdx_X;
                dzdx_P(:,:,:,i) = cur_dzdx_P;
            end
            
            derInputs{1} = dzdx_X;
            derInputs{2} = dzdx_P;
            derParams = {} ;            
        end
        
        function obj = meanshift_Y_is_XP(varargin)
            obj.load(varargin) ;
        end
    end
end
