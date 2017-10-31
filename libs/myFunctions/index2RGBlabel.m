function [ RGBlabel, evalLabelMap ]= index2RGBlabel(indexMap, colorLabel, classID)
%
% indexMap contains classID in [0~numClass-1], where numClass is size(colorLabel,1)
% colorLabel contains rgb values for all the numClass classes.
% 
% Shu Kong @uci
% 2/3/2017

%%
if nargin<3
    classID = zeros(1, size(colorLabel,1));
end

numClass = size(colorLabel,1);
R = zeros(size(indexMap));
G = zeros(size(indexMap));
B = zeros(size(indexMap));
evalLabelMap = zeros(size(indexMap));
for i = 0:numClass-1
    R(indexMap==i) = colorLabel(i+1,1);
    G(indexMap==i) = colorLabel(i+1,2);
    B(indexMap==i) = colorLabel(i+1,3);
    
    evalLabelMap(indexMap==i) = classID(i+1);
end
RGBlabel = cat(3, R,G,B);