function [sName] = addCombo_conv_BN_relu(net, sName, baseName, kernelSZ, hasBias, stride, pad, dilate)


lName = [baseName '_conv'];
block = dagnn.Conv('size', kernelSZ, 'hasBias', hasBias, 'stride', stride, 'pad', pad, 'dilate', dilate);
net.addLayer(lName, block, sName, lName, {[lName '_f']});
filter = randn(kernelSZ, 'single')*sqrt(2/kernelSZ(end));
net.params(net.layers(net.getLayerIndex(lName)).paramIndexes).value = filter;
net.params(net.layers(net.getLayerIndex(lName)).paramIndexes).weightDecay = 1;
net.params(net.layers(net.getLayerIndex(lName)).paramIndexes).learningRate = 10;
sName = lName;

lName = [baseName, '_bn'];
block = dagnn.BatchNorm('numChannels', kernelSZ(end));
% block.usingGlobal = false;
net.addLayer(lName, block, sName, lName, {[lName '_g'], [lName '_b'], [lName '_m']});
pidx = net.getParamIndex({[lName '_g'], [lName '_b'], [lName '_m']});
net.params(pidx(1)).weightDecay = 1;
net.params(pidx(2)).weightDecay = 1;
net.params(pidx(1)).learningRate = 10;
net.params(pidx(2)).learningRate = 10;
net.params(pidx(3)).learningRate = 0.1;
net.params(pidx(3)).trainMethod = 'average';
net.params(pidx(1)).value = ones([kernelSZ(end) 1], 'single'); % slope
net.params(pidx(2)).value = zeros([kernelSZ(end) 1], 'single');  % bias
net.params(pidx(3)).value = zeros([kernelSZ(end) 2], 'single'); % moments
sName = lName;

lName = [baseName, '_relu'];
block = dagnn.ReLU('leak', 0);
net.addLayer(lName, block, sName, lName);
sName = lName;


