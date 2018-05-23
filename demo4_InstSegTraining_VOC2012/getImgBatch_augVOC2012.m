function [imBatch, semanticMaskBatch, instanceMaskBatch, weightBatch] = getImgBatch_augVOC2012(images, mode, varargin)
% CNN_IMAGENET_GET_BATCH  Load, preprocess, and pack images for CNN evaluation
opts.imageSize = [56, 56] ;
opts.border = [10, 10] ;
opts.stepSize = [1, 1] ;
opts.lambda = 1 ;
opts.keepAspect = true ;
opts.numAugments = 0 ; % flip?
opts.transformation = 'none' ;  % 'stretch' 'none'
opts.averageImage = reshape( [123.6800 116.7800 103.9400], [1,1,3]) ;
% opts.rgbVariance = 1*ones(1,1,'single') ; % default: zeros(0,3,'single') ;
opts.rgbVariance = zeros(0,3,'single') ;
opts.classNum = 11;

opts.interpolation = 'bilinear' ;
opts.numThreads = 1 ;
opts.prefetch = false ;
opts.imdb = [];

opts = vl_argparse(opts, varargin);
%% read mat file for image and label
imBatch = zeros(opts.imageSize(1)-opts.border(1), opts.imageSize(2)-opts.border(2), 3, numel(images), 'single') ;
semanticMaskBatch = zeros((opts.imageSize(1)-opts.border(1)), (opts.imageSize(2)-opts.border(2)), 1, numel(images), 'single') ;
instanceMaskBatch = zeros((opts.imageSize(1)-opts.border(1)), (opts.imageSize(2)-opts.border(2)), 1, numel(images), 'single') ;
weightBatch = ones((opts.imageSize(1)-opts.border(1)), (opts.imageSize(2)-opts.border(2)), 1, numel(images), 'single') ;

gammaRange = [-0.03, 0.03];

for img_i = 1:numel(images)
    %% read the image and annotation
    if  strcmpi(mode, 'val')
        cur_path_to_image = fullfile(opts.imdb.path_to_image, opts.imdb.val_image{images(img_i)});
        cur_path_to_annot = fullfile(opts.imdb.path_to_annot, opts.imdb.val_annot{images(img_i)});
        flag_flip = 0;
    else
        cur_path_to_image = fullfile(opts.imdb.path_to_image, opts.imdb.train_image{images(img_i)});
        cur_path_to_annot = fullfile(opts.imdb.path_to_annot, opts.imdb.train_annot{images(img_i)});
        flag_flip = rand(1)>0.5;
    end
    
    cur_path_to_image = strrep(cur_path_to_image,'\','/');
    cur_path_to_annot = strrep(cur_path_to_annot,'\','/');
    cur_image = imread(cur_path_to_image);
    cur_segMap = load(cur_path_to_annot);
    cur_instMap = cur_segMap.gtMat.instMap;
    cur_segMap = cur_segMap.gtMat.segMap;
    
    cur_image = single(cur_image);
    cur_segMap = single(cur_segMap);
    cur_instMap = single(cur_instMap);
    
    weightBatch = ones(size(cur_segMap), 'single');
    imageSize = size(cur_image);
    border = opts.border;
    if  strcmpi(mode, 'val')
        xstart = 1;
        ystart = 1;
        xend = imageSize(2);
        yend = imageSize(1);
    else
        xstart = randperm(opts.border(2) / opts.stepSize(2) + 1,1)*opts.stepSize(2) - opts.stepSize(2) + 1;
        ystart = randperm(opts.border(1) / opts.stepSize(1) + 1,1)*opts.stepSize(1) - opts.stepSize(1) + 1;
        xend = imageSize(2) - (border(2) - xstart+1);
        yend = imageSize(1) - (border(1) - ystart+1);
    end
    %% augmentation
    if strcmpi(mode, 'train')
        if flag_flip
            %% flip augmentation
            cur_image = fliplr(cur_image);
            cur_segMap = fliplr(cur_segMap);
            cur_instMap = fliplr(cur_instMap);
        end
        %% crop augmentation
        cur_image = cur_image(ystart:yend, xstart:xend,:);
        cur_segMap = cur_segMap(ystart:yend, xstart:xend,:);
        cur_instMap = cur_instMap(ystart:yend, xstart:xend,:);
        %% gamma augmentation
        if rand(1)>0.3
            cur_image = cur_image / 255;
            Z = gammaRange(1) + (gammaRange(2)-gammaRange(1)).*rand(1);
            gamma = log(0.5 + 1/sqrt(2)*Z) / log(0.5 - 1/sqrt(2)*Z);
            cur_image = cur_image.^gamma * 255;
        end
        %% RGB jittering
        if rand(1)>0.3
            jitterRGB = rand(1,1,3)*0.4+0.8;
            cur_image = bsxfun(@times, cur_image, jitterRGB);
        end
        %% random rotation
        if rand(1)>0.3 && min(imageSize)>200
            rangeDegree = -15:1:15;
            angle = randsample(rangeDegree, 1);
            if angle~=0
                W = size(cur_image,2);
                H = size(cur_image,1);
                Hst = ceil(W*abs(sin(angle/180*pi)));
                Wst = ceil(H*abs(sin(angle/180*pi)));
                
                cur_image = imrotate(cur_image, angle, 'bicubic');
                cur_segMap = imrotate(cur_segMap, angle, 'nearest');
                cur_instMap = imrotate(cur_instMap, angle, 'nearest');
                
                cur_image = cur_image(max(1,Hst):end-Hst, max(1,Wst):end-Wst, :);
                cur_segMap = cur_segMap(max(1,Hst):end-Hst, max(1,Wst):end-Wst, :);
                cur_instMap = cur_instMap(max(1,Hst):end-Hst, max(1,Wst):end-Wst, :);
                
                cur_image(cur_image<0) = 0;
                cur_image(cur_image>255) = 255;
            end
        end
        %% random scaling
        sz = size(cur_image); sz = sz(1:2);
        if rand(1) > 0.3  && min(sz)>200
            scaleFactorList = 0.7:0.01:1.5;
            scaleFactorList = randsample(scaleFactorList, 1);
        else
            scaleFactorList = 1;
        end
        
        if max(scaleFactorList*sz)>800 % if too large to fit in memory
            tmp = 800 / max(scaleFactorList*sz);
            curRandScaleFactor = scaleFactorList*sz*tmp;
        else
            curRandScaleFactor = scaleFactorList*sz;
        end
        curRandScaleFactor = round(curRandScaleFactor/8)*8;
        
        cur_image = imresize(cur_image, curRandScaleFactor);
        cur_segMap = imresize(cur_segMap, curRandScaleFactor, 'nearest');
        cur_instMap = imresize(cur_instMap, curRandScaleFactor, 'nearest');
    elseif strcmpi(mode, 'val')
        %% crop augmentation
        sz = size(cur_image); sz = sz(1:2);
        curRandScaleFactor = round(1*sz/8)*8;
        cur_image = imresize(cur_image, curRandScaleFactor);
        cur_segMap = imresize(cur_segMap, curRandScaleFactor, 'nearest');
        cur_instMap = imresize(cur_instMap, curRandScaleFactor, 'nearest');
    end
    %% return
    imBatch = cur_image;
    imBatch = bsxfun(@minus, imBatch, opts.averageImage) ;
    semanticMaskBatch = cur_segMap +1; %[!!! background from value-0 to value 1 !!!], so there are 21 classes including background!!!
    instanceMaskBatch = cur_instMap;
end

% finishFlag = true;
%% leaving blank











