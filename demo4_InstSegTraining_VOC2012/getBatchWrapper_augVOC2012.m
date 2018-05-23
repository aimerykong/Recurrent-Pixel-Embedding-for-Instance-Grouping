% return a get batch function
% -------------------------------------------------------------------------
function fn = getBatchWrapper_augVOC2012(opts)
fn = @(images, mode) getBatch_dict(images, mode, opts) ;
end


function [imBatch, semanticMaskBatch, instanceMaskBatch, weightBatch] = getBatch_dict(images, mode, opts)
[imBatch, semanticMaskBatch, instanceMaskBatch, weightBatch] = getImgBatch_augVOC2012(images, mode, opts, 'prefetch', nargout == 0) ;
end
