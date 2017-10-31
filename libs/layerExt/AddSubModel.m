function net = AddSubModel(net, upsample_fac, bilinear_upsample, var_to_upsample_name, bases_size, num_basis, neigh_size, learningrate, upsample_2x_per_layer, opts)

up_name = [num2str(upsample_fac) 'x'];
perv_up_name = [num2str(2 * upsample_fac) 'x'];
ind_var = net.getVarIndex(var_to_upsample_name);
vsizes = net.getVarSizes({'input', [224 224 3 10]}); %,'label',[224 224 10]});
n_channels = vsizes{ind_var}(3);

net.addLayer(['bases_coef_' up_name], ...
    dagnn.Conv('size', [neigh_size neigh_size n_channels opts.num_classes*num_basis], 'pad', floor(neigh_size/2), 'hasBias', true), ...
    var_to_upsample_name, ['coef_' up_name], {['bases_coef_' up_name 'f'],['bases_coef_' up_name 'b']});

ind = net.getParamIndex(['bases_coef_' up_name 'f']);
net.params(ind).value = zeros(neigh_size, neigh_size, n_channels, opts.num_classes*num_basis, 'single');
net.params(ind).learningRate = learningrate;
net.params(ind).weightDecay = 1;
ind = net.getParamIndex(['bases_coef_' up_name 'b']);
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

rec_upsample = upsample_fac / bilinear_upsample;

net.addLayer(['deconv_' up_name], ...
    dagnn.ConvTranspose(...
    'size', size(filters), ...
    'upsample', rec_upsample, ...
    'crop', [rec_upsample/2 rec_upsample/2 rec_upsample/2 rec_upsample/2], ...
    'opts', {'cudnn','nocudnn'}, ...
    'numGroups', opts.num_classes, ...
    'hasBias', false), ...
    ['coef_' up_name], ['prediction_' up_name postname], ['deconv_' up_name 'f']) ;


f = net.getParamIndex(['deconv_' up_name 'f']);
net.params(f).value = filters;

net.params(f).learningRate = 0;
net.params(f).weightDecay = 1;

if upsample_fac < 32
    if upsample_2x_per_layer
        net = AddBilinearUpSampling(net, ['prediction_' perv_up_name], ['prediction_' perv_up_name '_bil_x2'], 2, opts);
        net.addLayer(['sum' up_name], dagnn.Sum(), {['prediction_' perv_up_name '_bil_x2'], ['prediction_' up_name '_add']}, ['prediction_' up_name]) ;
        if opts.dilate_erode_seg
            net = AddBilinearUpSampling(net, ['dil_seg' perv_up_name], ['dil_seg' perv_up_name '_bil_x2'], 2, opts);
            net.addLayer(['sum_dil' up_name], dagnn.Sum(), {['dil_seg' perv_up_name '_bil_x2'], ['dil_seg' up_name '_add']}, ['dil_seg' up_name]) ;
            
            net = AddBilinearUpSampling(net, ['ero_seg' perv_up_name], ['ero_seg' perv_up_name '_bil_x2'], 2, opts);
            net.addLayer(['sum_ero' up_name], dagnn.Sum(), {['ero_seg' perv_up_name '_bil_x2'], ['ero_seg' up_name '_add']}, ['ero_seg' up_name]) ;
        end
    else
        net.addLayer(['sum' up_name], dagnn.Sum(), {['prediction_' perv_up_name], ['prediction_' up_name '_add']}, ['prediction_' up_name]) ;
        
        if opts.dilate_erode_seg
            net.addLayer(['sum_dil' up_name], dagnn.Sum(), {['dil_seg' perv_up_name], ['dil_seg' up_name '_add']}, ['dil_seg' up_name]) ;
            net.addLayer(['sum_ero' up_name], dagnn.Sum(), {['ero_seg' perv_up_name], ['ero_seg' up_name '_add']}, ['ero_seg' up_name]) ;
        end
    end
end

prediction_var_name = ['prediction_' up_name];
post_label_name = num2str(bilinear_upsample);

% Add loss layer
net.addLayer(['objective_' up_name], ...
    SegmentationLoss('loss', 'softmaxlog'), ...
    {prediction_var_name, ['label' post_label_name]}, ['objective_' up_name]) ;

% Add accuracy layer
net.addLayer(['accuracy_' up_name], ...
    SegmentationAccuracy('numClasses', opts.num_classes), ...
    {prediction_var_name, ['label' post_label_name]}, ['accuracy_' up_name]) ;
