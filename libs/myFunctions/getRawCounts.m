function rawcounts = getRawCounts(gt, resim, numClass)
% Calculate the counts for prediction and ground truth..
num = numClass + 1;
rawcounts = zeros([num, num]);
locs = gt(:) >= 0;
sumim = 1+gt+resim*num;
hs = histc(sumim(locs), 1:num*num);
rawcounts = reshape(hs(:), size(rawcounts));
