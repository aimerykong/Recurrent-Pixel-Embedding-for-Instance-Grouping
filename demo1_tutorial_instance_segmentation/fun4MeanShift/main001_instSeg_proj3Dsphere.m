clc;
close all
%% which one to show?
idx = 24;
coordIndices = [1,2,3];

imgFig1 = figure(1);
set(imgFig1, 'Position', [100 100 1400 900]) % [1 1 width height]

subplot(2,2,1);
imagesc(imgMat(:,:,:,idx)); axis off image;
subplot(2,2,2);
imagesc(instanceMaskMat(:,:,:,idx)); axis off image;
subplot(2,2,3);
A = (predInstanceMaskMat0(:,:,coordIndices,idx) + 1) / 2;
imagesc(A);  axis off image;
%% 3D surface
subplot(2,2,4);

r = 1;
[x,y,z] = sphere(50);
x0 = 0; y0 = 0; z0 = 0;
x = x*r + x0;
y = y*r + y0;
z = z*r + z0;

% figure
lightGrey = 0.7*[1 1 1]; % It looks better if the lines are lighter
surface(x,y,z, 'FaceColor', 'none', 'EdgeColor',lightGrey)
hold on
%% points
hold on;
points = reshape(predInstanceMaskMat0(:,:,coordIndices,idx), [], 3);
points = points';
pointsColor = points - (-1);%min(points(:)); % -1
pointsColor = pointsColor ./ 2;%max(pointsColor(:));
for i = 1:size(points,2)
    plot3( points(1,i), points(2,i), points(3,i), 's', 'MarkerSize',3, 'MarkerFaceColor', pointsColor(:,i)', 'MarkerEdgeColor', pointsColor(:,i)');
end
hold off;

axis off square
view([1 1 0.75]) % adjust the viewing angle
zoom(1.4)

%% save result
if flagSaveFig    
    export_fig( sprintf('%s/%04d_visualization_single.jpg', saveFolder, i) );
end

