function feaMap = rescaleFeaMap(feaMap)
feaMap = feaMap - min(feaMap(:));
feaMap = feaMap ./ max(feaMap(:));
