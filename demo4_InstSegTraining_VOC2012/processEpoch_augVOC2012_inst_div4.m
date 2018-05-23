function [stats, prof] = processEpoch_augVOC2012_inst_div4(net, state, scalingFactor, opts, mode, totalEpoch)
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
        instanceMaskBatch = imresize(single(instanceMaskBatch), 1/4, 'nearest');
        %% visualization
        imBatch = gpuArray(imBatch) ;
        inputs = {'data', imBatch};
        %% fetch data for train/test
        if opts.withSemanticSeg
            inputs{end+1} = 'gt_div1_seg';
            inputs{end+1} = gpuArray(semanticMaskBatch);
        end
        if opts.withInstanceSeg
            inputs{end+1} = 'gt_ins_div4';
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
        end
        %% feedforward and/or backprop
        if strcmp(mode, 'train')
            net.mode = 'normal' ;
            net.accumulateParamDers = (s ~= 1) ;
            net.eval(inputs, opts.derOutputs, 'backPropAboveLayerName', opts.backPropAboveLayerName) ;            
        else
            %net.mode = 'test' ;
            net.mode = 'normal' ;
            net.eval(inputs) ;
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
