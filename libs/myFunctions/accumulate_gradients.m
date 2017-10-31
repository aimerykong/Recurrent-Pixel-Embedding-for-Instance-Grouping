function state = accumulate_gradients(state, net, opts, batchSize, mmap)
% -------------------------------------------------------------------------
numGpus = numel(opts.gpus) ;
otherGpus = setdiff(1:numGpus, labindex) ;

if ~strcmp(opts.backPropAboveLayerName, '')
    opts.backPropLayerAbove = net.getLayerIndex(opts.backPropAboveLayerName);
    opts.backPropParamAbove = net.layers( opts.backPropLayerAbove ).paramIndexes(1);
else
    opts.backPropLayerAbove = 1;
    opts.backPropParamAbove = 1;
end

for p=1:numel(net.params)
%     disp(p)
    if p < opts.backPropParamAbove
        continue;%return;
    end
    
    if isempty(net.params(p).der) && p > 102
        fprintf('\tempty der: %d\n', p);
    elseif isempty(net.params(p).der)
        continue;
    end
    % accumualte gradients from multiple labs (GPUs) if needed
    if numGpus > 1
        tag = net.params(p).name ;
        for g = otherGpus
            tmp = gpuArray(mmap.Data(g).(tag)) ;
            net.params(p).der = net.params(p).der + tmp ;
        end
    end   
    
    switch net.params(p).trainMethod        
        case 'average' % mainly for batch normalization
            %disp(p);
            thisLR = net.params(p).learningRate ;
            net.params(p).value = ...
                (1 - thisLR) * net.params(p).value + ...
                (thisLR/batchSize/net.params(p).fanout) * net.params(p).der ;
            
        case 'gradient'
            
            %elseif ~isempty(net.params(p).der)
                thisDecay = opts.weightDecay * net.params(p).weightDecay ;
                thisLR = state.learningRate * net.params(p).learningRate ;
                %             p
%                 if p==426
%                     disp(p)
%                 end
                state.momentum{p} = opts.momentum * state.momentum{p} ...
                    - thisDecay * net.params(p).value ...
                    - (1 / batchSize) * net.params(p).der ;
                net.params(p).value = net.params(p).value + thisLR * state.momentum{p} ;
            %end
        case 'otherwise'
            error('Unknown training method ''%s'' for parameter ''%s''.', ...
                net.params(p).trainMethod, ...
                net.params(p).name) ;
    end
end
