% return a get batch function
% -------------------------------------------------------------------------
function fn = getBatchWrapper4toyDigitV2(opts)
% -------------------------------------------------------------------------
    fn = @(images, mode) getBatch_dict4toyDigitV2(images, mode, opts) ;
end

% -------------------------------------------------------------------------
function [imBatch, semanticMaskBatch, instanceMaskBatch, weightBatch] = getBatch_dict4toyDigitV2(images, mode, opts)
% -------------------------------------------------------------------------
    %images = strcat([imdb.path_to_dataset filesep], imdb.(mode).(batch) ) ; 
    [imBatch, semanticMaskBatch, instanceMaskBatch, weightBatch] = getImgBatch4toyDigitV2(images, mode, opts, 'prefetch', nargout == 0) ;
end
