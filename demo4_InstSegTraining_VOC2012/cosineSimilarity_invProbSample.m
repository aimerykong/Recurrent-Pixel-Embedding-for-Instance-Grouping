classdef cosineSimilarity_invProbSample < dagnn.ElementWise
    properties
        randSampleRatio=0.1
    end
    properties
        sampledInput
        marginMat
        curSimMat
        weightMat
        numPixel
        numPixelPair
        pixelIndex
        pixelSub
        numInputs
        SIZE_
        mapping
    end
    
    methods
        function outputs = forward(obj, inputs, params)
            obj.numInputs = numel(inputs);
            X = inputs{1};
            % [height width channels batchsize]
            [h, w, ch, bs] = size(X);
            obj.SIZE_ = [h, w, ch, bs];
            
            % if GPU is used
            gpuMode = isa(X, 'gpuArray');
            obj.numPixel = floor(h*w*obj.randSampleRatio);
            obj.numPixel = min(obj.numPixel, 3500);
            
            obj.numPixelPair = obj.numPixel^2;
            
            if gpuMode
                obj.sampledInput = gpuArray(zeros(obj.numPixel, ch, 1, bs, 'single'));
                Y = gpuArray(zeros(obj.numPixel, obj.numPixel, 1, bs, 'single'));
                obj.curSimMat = gpuArray(zeros([obj.numPixel,obj.numPixel,1,bs], 'single'));
                obj.weightMat = gpuArray(zeros([obj.numPixel,obj.numPixel,1,bs], 'single'));
                obj.marginMat = gpuArray(zeros([obj.numPixel,obj.numPixel,1,bs], 'single'));
                obj.pixelIndex = gpuArray( zeros([obj.numPixel, 1, 1, bs], 'single') );
                obj.pixelSub = gpuArray( zeros([obj.numPixel, 2, 1, bs], 'single') );
            else
                obj.sampledInput = zeros(obj.numPixel, ch, 1, bs, 'single');
                Y = zeros(obj.numPixel, obj.numPixel, 1, bs, 'single');
                obj.curSimMat = zeros([obj.numPixel,obj.numPixel,1,bs], 'single');
                obj.weightMat = zeros([obj.numPixel,obj.numPixel,1,bs], 'single');
                obj.marginMat = zeros([obj.numPixel,obj.numPixel,1,bs], 'single');
                obj.pixelIndex = zeros([obj.numPixel, 1, 1, bs], 'single'); %
                obj.pixelSub = zeros([obj.numPixel, 2, 1, bs], 'single');
            end
            
            for j = 1:bs
                obj.pixelIndex(:,1,1,j) = squeeze(randperm(h*w, obj.numPixel));
                [obj.pixelSub(:,1,1,j), obj.pixelSub(:,2,1,j)] = ind2sub([h,w], obj.pixelIndex(:,1,1,j));
                
                curX = X(:,:,:,j);
                curX = reshape(curX, [], ch);
                curX = curX(obj.pixelIndex(:,1,1,j),:);
                
                C = inputs{2}(:,:,:,j);
                C = C(obj.pixelIndex(:,1,1,j));
                %C = C(:);
                W = C;
                
                marginMatTmp = [obj.pixelSub(:,1,1,j), obj.pixelSub(:,2,1,j)];
                mm = sum(marginMatTmp.^2,2);
                marginMatTmp = repmat(mm, 1, size(marginMatTmp,1)) + repmat(mm', size(marginMatTmp,1), 1) - 2*marginMatTmp*marginMatTmp';
                marginMatTmp = sqrt(marginMatTmp);
                marginMatTmp = marginMatTmp ./ max(marginMatTmp(:));
                %marginMatTmp = exp(-marginMatTmp/5);
                
                tmp_instanceIdx = unique(C);
                tmp_instanceNum = length(tmp_instanceIdx);
                for ii = 1:tmp_instanceNum
                    W(W==tmp_instanceIdx(ii)) = sum(C(:)==tmp_instanceIdx(ii));
                    %if tmp_instanceIdx(ii)==1
                    %    W(W==tmp_instanceIdx(ii)) = sum(C(:)==tmp_instanceIdx(ii))/2;
                    %else
                    %    W(W==tmp_instanceIdx(ii)) = sum(C(:)==tmp_instanceIdx(ii));
                    %end
                end
                W = W(:)*W(:)';
                W = 1 ./ W;
                W = W ./ sum(W(:));
                
                C = repmat(C, 1, numel(C));
                C = (C==C');
                
                Y(:,:,1,j) = curX*curX'*0.5 + 0.5;
                
                % inverse probability re-weighting
                invProbW = Y(:,:,1,j);
                invProbW = floor(invProbW*100)+1;
                invProbW(invProbW>100) = 100;
                invProbW(invProbW<0) = 1;
                invProbW = obj.mapping(invProbW);
                
                offset_factor = 10^4;
                
                obj.curSimMat(:,:,:,j) = C;
                obj.weightMat(:,:,:,j) = W .* invProbW * offset_factor;
                obj.marginMat(:,:,:,j) = marginMatTmp;
                
                obj.sampledInput(:,:,:,j) = curX;
                
                
            end
            
            outputs{1} = Y; % using euclidean loss
        end
        
        
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            derInputs = cell(1, numel(inputs));
            h = obj.SIZE_(1);
            w = obj.SIZE_(2);
            ch = obj.SIZE_(3);
            bs = obj.SIZE_(4);
            
            dzdy = derOutputs{1};
            X = obj.sampledInput; % inputs{1};
            gpuMode = isa(X, 'gpuArray');
            
            % do backward pass
            if gpuMode
                dzdx = gpuArray(zeros([h*w, ch, bs], 'single'));
            else
                dzdx = zeros([h*w, ch, bs], 'single');
            end
            
            for i = 1:obj.SIZE_(4) % obj.SIZE_ = [h, w, ch, bs]; -- [height width channels batchsize]
                dzdy_b = dzdy(:,:,:,i);
                %a = reshape(X(:,:,:,i), [h*w, ch]);
                %a = dzdy_b*a;
                a = dzdy_b*X(:,:,:,i);
                a = 2*a;
                
                tmp = dzdx(:,:,i);
                tmp(obj.pixelIndex(:,1,1,i),:) = a;
                dzdx(:,:,i) = tmp;
            end
            
            dzdx = reshape(dzdx, obj.SIZE_);
            derInputs{1} = dzdx ; % using euclidean loss;
            derParams = {} ;
        end
        
        function obj = cosineSimilarity_invProbSample(varargin)
            obj.load(varargin) ;
        end
    end
end
