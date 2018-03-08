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
            %% interpret
            %{
            X = inputs{1};
            X = permute(X, [3,1,2,4]);
            X = reshape(X, [size(X,1), prod(obj.SIZE_)/obj.SIZE_(3)] );
            sumX = sqrt(sum(X.^2,1));
            sumX = sumX + 0.00001;
            X = bsxfun(@rdivide, X, sumX);
            X = reshape(X, [size(X,1), obj.SIZE_(1:2), obj.SIZE_(4)]);
            X = permute(X, [2,3,1,4]);
            pred=X;
            
            figure(2);
            idx = 2;
            winH = 1;
            winW = 5;
            winIdx = 1;
            
            im_der = dzdx(:,20:end,:,idx);
            im = obj.net.vars(1).value(:,20:end,:,idx);
            gt = obj.net.vars(135).value(:,20:end,:,idx);
            gt_unique = unique(gt);
            for i = 1:length(gt_unique)
                gt(gt==gt_unique(i)) = -i;
            end
            gt = gt + length(gt_unique);
            pred = pred(:,20:end,:,idx);
            pred_show = pred + 1; pred_show = pred_show./2;
            pointColor_mapping = [0,0,0;
                1,0,0;
                0,1,0;
                0,0,1;
                0,1,1;
                1,0,1;
                1,1,0;
                0.5,0.5,0.5;
                1,0.5,0.5;
                0.5,1,0.5;
                0.5,0.5,1];
            
            pointsColor_gt = pointColor_mapping(gt(:)+1,:);
            pointsColor_gt = reshape(pointsColor_gt, [size(gt,1), size(gt,2), 3]);
            
            
            subplot(winH,winW,winIdx); winIdx = winIdx + 1;
            imshow(im); title('image');
            subplot(winH,winW,winIdx); winIdx = winIdx + 1;
            imagesc(gt); axis off image; title('gt-inst');
            subplot(winH,winW,winIdx); winIdx = winIdx + 1;
            imagesc(pointsColor_gt); axis off image; title('gt-inst color');
            subplot(winH,winW,winIdx); winIdx = winIdx + 1;
            imagesc(pred_show); axis off image; title('pred-inst');
            
            subplot(winH,winW,winIdx); winIdx = winIdx + 1;
%             A = im_der; A = A-min(A(:)); A = A./max(A(:));
            A = im_der; A = sum(A.^2,3);
            imagesc(A); axis off image; title('gradient magnitude'); colorbar;
            %% spherical visualization
            imgFig = figure(3);
            set(imgFig, 'Position', [100 100 1500 800]) % [1 1 width height]
            subplot(1,2,1); title('gradient with predicted color/embedding');
            
            r = 1;
            [x,y,z] = sphere(50);
            x0 = 0; y0 = 0; z0 = 0;
            x = x*r + x0;
            y = y*r + y0;
            z = z*r + z0;
            
            % figure
            lightGrey = 0.85*[1 1 1]; % It looks better if the lines are lighter
            surface(x,y,z, 'FaceColor', 'none', 'EdgeColor',lightGrey)
            hold on;
            
            coordIndices = [1,2,3];
            points = reshape(pred(:,:,coordIndices), [], 3);
            points = points';
            pointsColor = points - (-1);%min(points(:)); % -1
            pointsColor = pointsColor ./ 2;%max(pointsColor(:));
            for i = 1:size(points,2)
                plot3( points(1,i), points(2,i), points(3,i), 's', 'MarkerSize',2, 'MarkerFaceColor', pointsColor(:,i)', 'MarkerEdgeColor', pointsColor(:,i)');
            end
            hold on;
            
            %             axis off square
            %             view([-83 42]);
            %             view([1 1 0.75]) % adjust the viewing angle
            %             zoom(1.4)
            
            multiplier = 100;
            X = pred(:,:,1);
            Y = pred(:,:,2);
            Z = pred(:,:,3);
            U = im_der(:,:,1) * multiplier;
            V = im_der(:,:,2) * multiplier;
            W = im_der(:,:,3) * multiplier;
            q3 = quiver3(X,Y,Z,U,V,W, 0, 'linewidth',1);
            q3.Color = 'black';
            q3.MarkerSize = 1;
            axis off square
            view([-83 42]);
            %%
            subplot(1,2,2); title('gradient with gt color');
            r = 1;
            [x,y,z] = sphere(100);
            x0 = 0; y0 = 0; z0 = 0;
            x = x*r + x0;
            y = y*r + y0;
            z = z*r + z0;
            lightGrey = 0.85*[1 1 1]; % It looks better if the lines are lighter
            surface(x,y,z, 'FaceColor', 'none', 'EdgeColor',lightGrey)
            hold on;
            
            pointsColor_gt = reshape(pointsColor_gt, [], 3)';
            
            for i = 1:size(points,2)
                plot3( points(1,i), points(2,i), points(3,i), 's', 'MarkerSize', 2, 'MarkerFaceColor', pointsColor_gt(:,i)', 'MarkerEdgeColor', pointsColor_gt(:,i)');
            end
            
            multiplier = 100;
            X = pred(:,:,1);
            Y = pred(:,:,2);
            Z = pred(:,:,3);
            U = im_der(:,:,1) * multiplier;
            V = im_der(:,:,2) * multiplier;
            W = im_der(:,:,3) * multiplier;
            q3 = quiver3(X,Y,Z,U,V,W, 0, 'linewidth',1);
            q3.Color = 'black';
            q3.MarkerSize = 1;
            axis off square
            view([-83 42]);
            disp(stop);
            %}
        end
        
        function obj = L2normalization(varargin)
            obj.load(varargin) ;
        end
    end
end
