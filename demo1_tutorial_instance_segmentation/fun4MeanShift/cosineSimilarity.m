classdef cosineSimilarity < dagnn.ElementWise
    properties (Transient)
        numInputs
        SIZE_
    end
    
    methods        
        function outputs = forward(obj, inputs, params)
            obj.numInputs = numel(inputs);
            X = inputs{1};           
            
            % if GPU is used
            gpuMode = isa(X, 'gpuArray');

            % [height width channels batchsize]
            [h, w, ch, bs] = size(X);
            obj.SIZE_ = [h, w, ch, bs];
            if gpuMode
                Y = gpuArray(zeros(h*w, h*w, 1, bs, 'single'));
            else
                Y = zeros(h*w, h*w, 1, bs, 'single');
            end
            
            for i = 1:bs
                curX = reshape(X(:,:,:,i), [h*w, ch]);
                Y(:,:,1,i) = curX*curX';
            end
                        
            %outputs{1} = (Y+1.1) / 2.2; % is using cross-entropy loss
            outputs{1} = Y; % using euclidean loss
        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            derInputs = cell(1, numel(inputs));
            h = obj.SIZE_(1);
            w = obj.SIZE_(2);
            ch = obj.SIZE_(3);
            
            dzdy = derOutputs{1};            
            X = inputs{1};
            gpuMode = isa(X, 'gpuArray');
            
            % do backward pass
            if gpuMode
                dzdx = gpuArray(zeros(size(X), 'single'));
            else
                dzdx = zeros(size(X), 'single');
            end
            
            for i = 1:obj.SIZE_(4) % obj.SIZE_ = [h, w, ch, bs]; -- [height width channels batchsize]
                dzdy_b = dzdy(:,:,:,i);               
                a = reshape(X(:,:,:,i), [h*w, ch]);                
                a = dzdy_b*a;
                dzdx(:, :, :, i) = 2*reshape(a, [h, w, ch]);
                
                % yang: bug here, should be 2* the original
                %Y(:, :, :, b) = 2*reshape(a * dzdy_b, [h, w, ch]);                
            end
            %derInputs{1} = dzdx / 2.2;  % is using cross-entropy loss
            derInputs{1} = dzdx ; % using euclidean loss;
            derParams = {} ;            
        end
        
        function obj = Scale(varargin)
            obj.load(varargin) ;
        end
    end
end
