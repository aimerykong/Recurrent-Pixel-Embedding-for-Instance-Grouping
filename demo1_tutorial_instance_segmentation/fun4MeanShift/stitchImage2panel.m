function showcaseSamples = stitchImage2panel(imgMat, panelSZ, M, N, st, ed)
% num = 25; % the total number of example to show
% M = 5; % the number of rows in the panel, each row shows N images as below
% N = 5; % the number of columns in the panel, each column shows M images as above
% panelSZ = round(64/2)*2; % the size (height/width) of the small square image


showcaseSamples_R = imgMat(:,:,1,st:ed);
showcaseSamples_R = reshape(showcaseSamples_R, [numel(showcaseSamples_R(:,:,1,1)), ed]);
showcaseSamples_R = showStitch(showcaseSamples_R, [panelSZ panelSZ], M, N, 'whitelines', 'linewidth', 5);
showcaseSamples_G = imgMat(:,:,2,st:ed);
showcaseSamples_G = reshape(showcaseSamples_G, [numel(showcaseSamples_G(:,:,1,1)), ed]);
showcaseSamples_G = showStitch(showcaseSamples_G, [panelSZ panelSZ], M, N, 'whitelines', 'linewidth', 5);
showcaseSamples_B = imgMat(:,:,3,st:ed);
showcaseSamples_B = reshape(showcaseSamples_B, [numel(showcaseSamples_B(:,:,1,1)), ed]);
showcaseSamples_B = showStitch(showcaseSamples_B, [panelSZ panelSZ], M, N, 'whitelines', 'linewidth', 5);
showcaseSamples = cat(3, showcaseSamples_R, showcaseSamples_G, showcaseSamples_B);