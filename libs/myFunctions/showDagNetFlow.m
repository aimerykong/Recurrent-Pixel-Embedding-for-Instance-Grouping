function showDagNetFlow(netbasemodel)
% show information flow within the architecture
%
% Shu Kong
% 08/09/2016

%%
for i = 1:numel(netbasemodel.layers)
    fprintf('layer-%03d %s -- ', i, netbasemodel.layers(i).name );
    
    %% input
    fprintf('\n\tinput:\n');
    for j = 1:length(netbasemodel.layers(i).inputIndexes)
        fprintf('\t\t%s\n', netbasemodel.vars(netbasemodel.layers(i).inputIndexes(j)).name);
    end
    %% output
    fprintf('\toutput:\n');
    for j = 1:length(netbasemodel.layers(i).outputIndexes)
        fprintf('\t\t%s\n', netbasemodel.vars(netbasemodel.layers(i).outputIndexes(j)).name);
    end
end
