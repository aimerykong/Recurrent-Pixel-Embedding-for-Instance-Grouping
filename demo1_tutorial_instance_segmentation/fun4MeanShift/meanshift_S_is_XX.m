classdef meanshift_S_is_XX < dagnn.ElementWise
    properties (Transient)
        numInputs
        SIZE_
        analyzeGradient=false
    end
    
    % [TODO]: current version only supports batchSize=1; need to extend to 
    % multiple input images    
    methods        
        function outputs = forward(obj, inputs, params)
            obj.numInputs = numel(inputs);
            assert(obj.numInputs == 1);
            X = inputs{1}; % dim x Num
            gpuMode = isa(X, 'gpuArray');
            obj.SIZE_ = size(X);            
            % [height width channels batchsize]
            [h, w, ch, bs] = size(X);
            if gpuMode
                Y = gpuArray(zeros(h*w, h*w, 1, bs, 'single'));
            else
                Y = zeros(h*w, h*w, 1, bs, 'single');
            end
            for i = 1:bs
                cur_X = X(:,:,:,i);
                cur_X = reshape(cur_X, [h*w, ch]);
                Y(:,:,:,i) = cur_X*cur_X';
            end            
            outputs{1} = Y;
        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            %%
            derInputs = cell(1, numel(inputs));
            X = inputs{1}; 
            gpuMode = isa(X, 'gpuArray');                        
            dzdy = derOutputs{1};               
            [h, w, ch, bs] = size(X);
            if gpuMode
                dzdx = gpuArray(zeros(h, w, ch, bs, 'single'));
            else
                dzdx = zeros(h, w, ch, bs, 'single');
            end
            
            for i = 1:bs
                cur_X = X(:,:,:,i);
                cur_X = reshape(cur_X, [h*w, ch]);
                cur_dzdx = 2*cur_X'*dzdy(:,:,:,i);
                cur_dzdx = reshape(cur_dzdx', [obj.SIZE_(1), obj.SIZE_(2), obj.SIZE_(3)]);
                dzdx(:,:,:,i) = cur_dzdx;
            end
            derInputs{1} = dzdx;
            derParams = {} ;         
            
            
            if obj.analyzeGradient
                %% visualize gradient and points
                %             for idx = 1:5
                %                 subplot(2,3,idx);
                %                 A = inputs{1}(:,:,:,idx);  A = A - min(A(:)); A = A./max(A(:)); imagesc(A); axis off image;
                %             end
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
                pred = inputs{1}(:,20:end,:,idx);
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
                %A = im_der; A = A-min(A(:)); A = A./max(A(:));
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
            end
        end
        
        function obj = meanshift_S_is_XX(varargin)
            obj.load(varargin) ;
        end
    end
end
