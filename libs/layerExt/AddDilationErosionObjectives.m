function net = AddDilationErosionObjectives(net, upsample_fac, rec_upsample, var_to_upsample, bases_size, num_basis, neigh_size, learningrate, opts)
up_name = [num2str(upsample_fac) 'x'];
net = AddSegObjective(net, var_to_upsample, up_name, upsample_fac, upsample_fac/rec_upsample, rec_upsample, neigh_size, num_basis, bases_size, 'dil', learningrate, opts);
net = AddSegObjective(net, var_to_upsample, up_name, upsample_fac, upsample_fac/rec_upsample, rec_upsample, neigh_size, num_basis, bases_size, 'ero', learningrate, opts);

function net = AddSegObjective(net, var_to_upsample, up_name, upsample_fac, bilinear_upsample, rec_upsample, neigh_size, num_basis, bases_size, sub_name, learningrate, opts)

ind_var = net.getVarIndex(var_to_upsample);
vsizes = net.getVarSizes({'input', [224 224 3 10]});
n_channels = vsizes{ind_var}(3);

conv_name = [sub_name '_seg' up_name '_coef'];
net.addLayer(conv_name, ...
    dagnn.Conv('size', [neigh_size neigh_size n_channels opts.num_classes*num_basis], 'pad', floor(neigh_size/2), 'hasBias', true), ...
    var_to_upsample, conv_name, {[conv_name 'f'],[conv_name 'b']});

ind = net.getParamIndex([conv_name 'f']);
net.params(ind).value = zeros(neigh_size, neigh_size, n_channels, opts.num_classes*num_basis, 'single');

net.params(ind).learningRate = learningrate;
net.params(ind).weightDecay = 1;

ind = net.getParamIndex([conv_name 'b']);
net.params(ind).value = zeros([1 opts.num_classes*num_basis], 'single');

net.params(ind).learningRate = 2;
net.params(ind).weightDecay = 1;

load(opts.bases_add);
assert(size(f,4) == opts.num_classes * num_basis);
if size(f,1) ~= bases_size
    fr = zeros(bases_size, bases_size, size(f,3), size(f,4));
    for fi = 1 : size(f, 4)
        fr(:,:,:,fi) = imresize(f(:,:,:,fi), bases_size/size(f,1));
    end
    f = fr;
end
filters = single(f);

postname = '_add';
if upsample_fac == 32
    postname = '';
end

deconv_name = [sub_name '_seg_deconv_' up_name postname];
type_name_ = [sub_name '_seg' up_name];
type_name = [type_name_ postname];
net.addLayer(deconv_name, ...
    dagnn.ConvTranspose(...
    'size', size(filters), ...
    'upsample', rec_upsample, ...
    'crop', [rec_upsample/2 rec_upsample/2 rec_upsample/2 rec_upsample/2], ...
    'opts', {'cudnn','nocudnn'}, ...
    'numGroups', opts.num_classes, ...
    'hasBias', true), ...
    conv_name, type_name, {[deconv_name 'f'], [deconv_name 'b']}) ;

ind = net.getParamIndex([deconv_name 'f']);
net.params(ind).value = filters;
net.params(ind).learningRate = 0;
net.params(ind).weightDecay = 1;

ind = net.getParamIndex([deconv_name 'b']);
net.params(ind).value = -ones([1 opts.num_classes], 'single');
net.params(ind).value(1) = 1;
net.params(ind).learningRate = 2;
net.params(ind).weightDecay = 1;

obj_name = ['obj_' sub_name '_seg' up_name];
net.addLayer(obj_name, ...
    SegmentationLossLogistic('loss', 'logistic'), ...
    {type_name_, [sub_name '_gt_' num2str(bilinear_upsample)]}, obj_name) ;
