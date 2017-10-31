function net = LLRAddMasking(net, upsample_fac, bilinear_up, upsample_2x_per_layer)

if upsample_2x_per_layer
    post_name = '_bil_x2';
else
    post_name = '';
end
up_name = [num2str(upsample_fac) 'x'];
pre_up_name = [num2str(2 * upsample_fac) 'x'];


pool_size = (upsample_fac * 2) / bilinear_up + 1;
assert(pool_size == 9);
pad_size = floor(pool_size)/2;
% Dilation of class probabilities
net.addLayer(['prob_' pre_up_name], dagnn.SoftMax(), ['prediction_' pre_up_name post_name], ['prob_' pre_up_name], {});
net.addLayer(['prob_dilate_' pre_up_name], dagnn.Pooling('stride', [1 1], 'poolSize', [pool_size pool_size], ...
    'pad', [pad_size, pad_size, pad_size, pad_size]), ['prob_' pre_up_name], ['prob_' pre_up_name '_dilate'], {});

% Dilation of negative of class probabilities
net.addLayer(['neg_prob_' pre_up_name], Neg(), ['prob_' pre_up_name], ['neg_prob_' pre_up_name], {});
['neg_prob_' pre_up_name]
net.addLayer(['neg_prob_dilate_' pre_up_name], dagnn.Pooling('stride', [1 1], 'poolSize', [pool_size pool_size], ...
    'pad', [pad_size, pad_size, pad_size, pad_size]), ['neg_prob_' pre_up_name], ['neg_prob_' pre_up_name '_dilate'], {});

% Sum of two dilation
net.addLayer(['bound_mask' pre_up_name], dagnn.Sum(), {['prob_' pre_up_name '_dilate'], ['neg_prob_' pre_up_name '_dilate']}, ['bound_mask' pre_up_name]) ;
net.addLayer(['dot_prod_' up_name], DotProduct(), {['bound_mask' pre_up_name], ['prediction_' up_name '_add']}, ['pred_' up_name '_aft_DP']) ;
net.setLayerInputs(['sum' up_name], {['prediction_' pre_up_name post_name], ['pred_' up_name '_aft_DP']});


