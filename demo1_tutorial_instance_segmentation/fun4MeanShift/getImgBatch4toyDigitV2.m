function [imBatch, semanticMaskBatch, instanceMaskBatch, weightBatch] = getImgBatch4toyDigitV2(images, mode, varargin)
% CNN_IMAGENET_GET_BATCH  Load, preprocess, and pack images for CNN evaluation
opts.imageSize = [56, 56] ;
opts.border = [0, 0] ;
opts.stepSize = [1, 1] ;
opts.lambda = 1 ;
opts.keepAspect = true ;
opts.numAugments = 0 ; % flip?
opts.transformation = 'none' ;  % 'stretch' 'none'
opts.averageImage = [] ;
% opts.rgbVariance = 1*ones(1,1,'single') ; % default: zeros(0,3,'single') ;
opts.rgbVariance = zeros(0,3,'single') ;
opts.classNum = 11;

opts.interpolation = 'bilinear' ;
opts.numThreads = 1 ;
opts.prefetch = false ;
opts.imdb = [];

opts = vl_argparse(opts, varargin);
% global dataset
%% read mat (hdf5) file for image and label
imBatch = zeros(opts.imageSize(1)-opts.border(1), opts.imageSize(2)-opts.border(2), 3, numel(images), 'single') ;
semanticMaskBatch = zeros((opts.imageSize(1)-opts.border(1)), (opts.imageSize(2)-opts.border(2)), 1, numel(images), 'single') ;
instanceMaskBatch = zeros((opts.imageSize(1)-opts.border(1)), (opts.imageSize(2)-opts.border(2)), 1, numel(images), 'single') ;
weightBatch = ones((opts.imageSize(1)-opts.border(1)), (opts.imageSize(2)-opts.border(2)), 1, numel(images), 'single') ;

for img_i = 1:numel(images)
    %% read the image and annotation    
    curMat = load(fullfile(opts.imdb.path_to_dataset, images{img_i}));
    imBatch = curMat.imgMat;
    semanticMaskBatch = curMat.semanticMaskMat;
    instanceMaskBatch = curMat.instanceMaskMat;             
    weightBatch = ones(size(instanceMaskBatch), 'single');
%     figure(1);
%     subplot(2,2,1); imshow(uint8(imgOrg));
%     subplot(2,2,2); imagesc(gtOrg); axis off image; caxis([0, 38]);
%     subplot(2,2,3); imagesc(gtDepthOrg); axis off image;
%     mask = gtDepthOrg>1000;
%     subplot(2,2,4); imagesc(mask); axis off image;
    %%
    %{
    if  strcmpi(mode, 'val')        
        flag_flip = 0;
        xstart = 1;
        ystart = 1;
        
        xend = opts.imageSize(2);
        yend = opts.imageSize(1);
    else
        flag_flip = rand(1)>0.5;
        xstart = randperm(opts.border(2) / opts.stepSize(2) + 1,1)*opts.stepSize(2) - opts.stepSize(2) + 1;
        ystart = randperm(opts.border(1) / opts.stepSize(1) + 1,1)*opts.stepSize(1) - opts.stepSize(1) + 1;
        
        xend = opts.imageSize(2) - (opts.border(2) - xstart+1);
        yend = opts.imageSize(1) - (opts.border(1) - ystart+1);
    end
    
    %% augmentation
    if strcmpi(mode, 'train') 
        if flag_flip
            %% flip augmentation
            imgOrg = fliplr(imgOrg);
            gtOrg = fliplr(gtOrg);
            gtDepthOrg = fliplr(gtDepthOrg);
        end        
        %% crop augmentation
        imgOrg = imgOrg(ystart:yend, xstart:xend,:);              
        gtOrg = gtOrg(ystart:yend, xstart:xend,:);    
        gtDepthOrg = gtDepthOrg(ystart:yend, xstart:xend,:);
        %% gamma augmentation
        imgOrg = imgOrg / 255;
        Z = gammaRange(1) + (gammaRange(2)-gammaRange(1)).*rand(1);
        gamma = log(0.5 + 1/sqrt(2)*Z) / log(0.5 - 1/sqrt(2)*Z);
        imgOrg = imgOrg.^gamma * 255;
        %% RGB jittering
        if rand(1)>0.5
            jitterRGB = rand(1,1,3)*0.4+0.8;
            imgOrg = bsxfun(@times, imgOrg, jitterRGB);            
        end
        %% random rotation
        if rand(1)>0.5
            rangeDegree = -10:1:10;
            angle = randsample(rangeDegree, 1);
            if angle~=0
                W = size(imgOrg,2);
                H = size(imgOrg,1);
                Hst = ceil(W*abs(sin(angle/180*pi)));
                Wst = ceil(H*abs(sin(angle/180*pi)));
                
                imgOrg = imrotate(imgOrg, angle, 'bicubic');
                gtOrg = imrotate(gtOrg, angle, 'nearest');
                gtDepthOrg = imrotate(gtDepthOrg, angle, 'bicubic');
                
                imgOrg = imgOrg(max(1,Hst):end-Hst, max(1,Wst):end-Wst, :);
                gtOrg = gtOrg(max(1,Hst):end-Hst, max(1,Wst):end-Wst, :);
                gtDepthOrg = gtDepthOrg(max(1,Hst):end-Hst, max(1,Wst):end-Wst, :);
                
                imgOrg(imgOrg<0) = 0;
                imgOrg(imgOrg>255) = 255;
            end
        %% random scaling          
            sz = size(imgOrg); sz = sz(1:2);
            scaleFactorList = 0.5:0.01:2;
            scaleFactorList = randsample(scaleFactorList, 1);
            curRandScaleFactor = round(scaleFactorList*sz/8)*8;
            
            imgOrg = imresize(imgOrg, curRandScaleFactor);
            gtOrg = imresize(gtOrg, curRandScaleFactor, 'nearest');
            gtDepthOrg = imresize(gtDepthOrg, curRandScaleFactor, 'bicubic');
            gtDepthOrg = gtDepthOrg ./ scaleFactorList;
            
%             mask = (gtOrg~=0);
        end
    elseif strcmpi(mode, 'val') 
        %% crop augmentation
        imgOrg = imgOrg(ystart:yend, xstart:xend,:);    
        gtOrg = gtOrg(ystart:yend, xstart:xend,:);                
        gtDepthOrg = gtDepthOrg(ystart:yend, xstart:xend,:);       
    end   
 %}
    %% return
    imBatch = bsxfun(@minus, imBatch, opts.averageImage) ;   
end
% imBatch = reshape(imBatch, [size(imBatch,1), size(imBatch,2), 1, size(imBatch,3)]);
semanticMaskBatch = reshape(semanticMaskBatch, [size(semanticMaskBatch,1), size(semanticMaskBatch,2), 1, size(semanticMaskBatch,3)]);
instanceMaskBatch = reshape(instanceMaskBatch, [size(instanceMaskBatch,1), size(instanceMaskBatch,2), 1, size(instanceMaskBatch,3)]);
weightBatch = reshape(weightBatch, [size(weightBatch,1), size(weightBatch,2), 1, size(weightBatch,3)]);



if  strcmpi(mode, 'val')
    a = 1:size(imBatch,4);
    imBatch = imBatch(:,:,:,a);
    semanticMaskBatch = semanticMaskBatch(:,:,:,a);
    instanceMaskBatch = instanceMaskBatch(:,:,:,a);
    weightBatch = weightBatch(:,:,:,a);
    
else
    a = randperm(size(imBatch,4), 5);
    imBatch = imBatch(:,:,:,a);
    semanticMaskBatch = semanticMaskBatch(:,:,:,a);
    instanceMaskBatch = instanceMaskBatch(:,:,:,a);
    weightBatch = weightBatch(:,:,:,a);
end

a = find(semanticMaskBatch==0);
semanticMaskBatch(a) = 11;
instanceMaskBatch(a) = 11;

% finishFlag = true;
%% leaving blank


