function [stats, prof] = processEpoch(net, state, scalingFactor, opts, mode, totalEpoch)
% -------------------------------------------------------------------------

%% initialize empty momentum
if strcmp(mode,'train')
    state.momentum = num2cell(zeros(1, numel(net.params))) ;
end

%% move CNN  to GPU as needed
numGpus = numel(opts.gpus) ;
if numGpus >= 1
    net.move('gpu') ;
    if strcmp(mode,'train')
        state.momentum = cellfun(@gpuArray,state.momentum,'UniformOutput',false) ;
    end
end
if numGpus > 1
    mmap = map_gradients(opts.memoryMapFile, net, numGpus) ;
else
    mmap = [] ;
end

%% profile
if opts.profile
    if numGpus <= 1
        profile clear ;
        profile on ;
    else
        mpiprofile reset ;
        mpiprofile on ;
    end
end

subset = state.(mode) ;
num = 0 ;
stats.num = 0 ; % return something even if subset = []
stats.time = 0 ;
adjustTime = 0 ;

start = tic ;
for t=1:opts.batchSize:numel(subset)
    fprintf('%s: epoch %02d/%03d: %3d/%3d:', mode, state.epoch, totalEpoch, ...
        fix((t-1)/opts.batchSize)+1, ceil(numel(subset)/opts.batchSize)) ;
    batchSize = min(opts.batchSize, numel(subset) - t + 1) ;
    
    for s=1:opts.numSubBatches
        % get this image batch and prefetch the next
        batchStart = t + (labindex-1) + (s-1) * numlabs ;
        batchEnd = min(t+opts.batchSize-1, numel(subset)) ;
        batch = subset(batchStart : opts.numSubBatches * numlabs : batchEnd) ;
        num = num + numel(batch) ;
        if numel(batch) == 0, continue ; end
                
        [imBatch, semanticMaskBatch, instanceMaskBatch, weightBatch] = state.getBatch(batch, mode) ;
        %% visualization
        %{
        num = 200;
        M = 20;
        N = 10;
        
        showcaseSamples = squeeze(imBatch);
        showcaseSamples = reshape(showcaseSamples, [numel(showcaseSamples(:,:,1)), num]);
        showcaseSamples = showStitch(showcaseSamples, 2*[28 28], M, N, 'linewidth', 1);
        
        showcaseSemanticMask = squeeze(semanticMaskBatch);
        showcaseSemanticMask = reshape(showcaseSemanticMask, [numel(showcaseSemanticMask(:,:,1)), num]);
        showcaseSemanticMask = showStitch(showcaseSemanticMask, 2*[28 28], M, N);
        
        showcaseInstanceMask = squeeze(instanceMaskBatch);
        showcaseInstanceMask = reshape(showcaseInstanceMask, [numel(showcaseInstanceMask(:,:,1)), num]);
        showcaseInstanceMask = showStitch(showcaseInstanceMask, 2*[28 28], M, N);
        
        imgFig2 = figure(2);
        subWindowH = 1;
        subWindowW = 3;
        set(imgFig2, 'Position', [100 100 1600 700]) % [1 1 width height]
        windowID = 1;
        subplot(subWindowH, subWindowW, windowID); imagesc(showcaseSamples); axis off image; title('random samples'); colorbar; windowID = windowID + 1;
        subplot(subWindowH, subWindowW, windowID); imagesc(showcaseSemanticMask); axis off image; title('showcase SemanticMask'); caxis([0, 11]); colorbar; windowID = windowID + 1;
        subplot(subWindowH, subWindowW, windowID); imagesc(showcaseInstanceMask); axis off image; title('showcase instanceMask'); colorbar; windowID = windowID + 1;
        %}        
        imBatch = gpuArray(imBatch) ;
        inputs = {'input', imBatch};
        %% fetch data for train/test        
%         figure(1); subplot(2,3,1); imagesc(uint8(imo)); axis off image;
        if opts.withSemanticSeg             
            inputs{end+1} = 'gt_seg';
            inputs{end+1} = gpuArray(semanticMaskBatch);
        end                
        if opts.withInstanceSeg 
            inputs{end+1} = 'gt_ins';
            inputs{end+1} = gpuArray(instanceMaskBatch);
        end 
        if opts.withWeights 
            inputs{end+1} = 'gt_weights';
            inputs{end+1} = gpuArray(weightBatch);
        end 
                
        if opts.prefetch
            if s == opts.numSubBatches
                batchStart = t + (labindex-1) + opts.batchSize ;
                batchEnd = min(t+2*opts.batchSize-1, numel(subset)) ;
            else
                batchStart = batchStart + numlabs ;
            end
            nextBatch = subset(batchStart : opts.numSubBatches * numlabs : batchEnd) ;
            state.getBatch(nextBatch, mode, scalingFactor) ;
%             state.getBatch(state.imdb, nextBatch) ;
        end
        %% feedforward and/or backprop
        if strcmp(mode, 'train')
%             net.mode = lower('trainGlobalBN') ; % trainGlobalBN, trainLocalBN, test, normal
            net.mode = 'normal' ;
            net.accumulateParamDers = (s ~= 1) ;
            
%             outputIdx = net.layers(net.getLayerIndex('res7_interp')).outputIndexes;
%             net.vars(outputIdx).precious = 1;            
%             net.eval(inputs, opts.derOutputs) ;
            net.eval(inputs, opts.derOutputs, 'backPropAboveLayerName', opts.backPropAboveLayerName) ;
           %{
            img = inputs{2};
            img = bsxfun(@plus, img, net.meta.normalization.averageImage);
            
            labelDict = load('labelDictionary.mat');
            validIndex = find(labelDict.ignoreInEval==0);
            colorLabel = labelDict.colorLabel(validIndex,:);       
            
            pred = gather(net.vars(outputIdx).value);
            [~, pred] = max(pred, [], 3);
            pred = index2RGBlabel(pred-1, colorLabel);
            
            gt = inputs{4};
%             [~, gt] = max(gt, [], 3);
            gt = index2RGBlabel(gt-1, colorLabel);
            
            figure; 
            subplot(1,3,1); imshow(uint8(img)); title('image');
            subplot(1,3,2); imshow(uint8(gt)); title('gt');
            subplot(1,3,3); imshow(uint8(pred)); title('pred');
            disp('done');
            %}
        else
            net.mode = 'test' ;
            net.mode = 'normal' ;
%             net.conserveMemory = 0;
%             outputIdx = net.layers(net.getLayerIndex('res7_interp')).outputIndexes;
%             net.vars(outputIdx).precious = 1;
            net.eval(inputs) ;
            %{
            img = inputs{2};
            img = bsxfun(@plus, img, net.meta.normalization.averageImage);
            
            labelDict = load('labelDictionary.mat');
            validIndex = find(labelDict.ignoreInEval==0);
            colorLabel = labelDict.colorLabel(validIndex,:);       
            
%             outputIdx = net.layers(net.getLayerIndex('res7_interp')).outputIndexes;
            pred = gather(net.vars(outputIdx).value);
            [~, pred] = max(pred, [], 3);
            pred = index2RGBlabel(pred-1, colorLabel);
            
            gt = inputs{4};
%             [~, gt] = max(gt, [], 3);
            gt = index2RGBlabel(gt-1, colorLabel);
            
            figure; 
            subplot(1,3,1); imshow(uint8(img)); title('image');
            subplot(1,3,2); imshow(uint8(gt)); title('gt');
            subplot(1,3,3); imshow(uint8(pred)); title('pred');
            disp('done');
            %}
        end
    end
    
    %% accumulate gradient
    if strcmp(mode, 'train')
        if ~isempty(mmap)
            write_gradients(mmap, net) ;
            labBarrier() ;
        end
        state = accumulate_gradients(state, net, opts, batchSize, mmap) ;
    end
    
    % get statistics
    time = toc(start) + adjustTime ;
    batchTime = time - stats.time ;
    stats = opts.extractStatsFn(net) ;
    stats.num = num ;
    stats.time = time ;
    currentSpeed = batchSize / batchTime ;
    averageSpeed = (t + batchSize - 1) / time ;
    if t == opts.batchSize + 1
        % compensate for the first iteration, which is an outlier
        adjustTime = 2*batchTime - time ;
        stats.time = time + adjustTime ;
    end
    
    fprintf(' %.1f (%.1f) Hz', averageSpeed, currentSpeed) ;
    for f = setdiff(fieldnames(stats)', {'num', 'time'})
        f = char(f) ;
        fprintf(' %s:', f) ;
        fprintf(' %.4f', stats.(f)(1)) ;        
    end
    fprintf('\n') ;
end

if ~isempty(mmap)
    unmap_gradients(mmap) ;
end

if opts.profile
    if numGpus <= 1
        prof = profile('info') ;
        profile off ;
    else
        prof = mpiprofile('info');
        mpiprofile off ;
    end
else
    prof = [] ;
end

net.reset() ;
net.move('cpu') ;
