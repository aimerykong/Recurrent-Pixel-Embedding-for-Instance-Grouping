function plotLearningCurves(stats)

if ~isempty(stats) && isfield(stats, 'val') && ~isempty(stats.val)
    plots = setdiff(...
        cat(2,...
        fieldnames(stats.train)', ...
        fieldnames(stats.val)'), {'num', 'time'}) ;
    for p = plots
        p = char(p) ;
        values = zeros(0, length(stats.train)) ;
        leg = {} ;
        for f = {'train', 'val'}
            f = char(f) ;
            if isfield(stats.(f), p)
                tmp = [stats.(f).(p)] ;
                values(end+1,:) = tmp(1,:)' ;
                leg{end+1} = f ;
            end
        end
        if numel(plots)<=2
            subplot(1,numel(plots),find(strcmp(p,plots))) ;
        elseif numel(plots) <= 4 
            subplot(2, 2, find(strcmp(p,plots))) ;
        elseif numel(plots) <= 6 
            subplot(2, 3, find(strcmp(p,plots))) ;
        elseif numel(plots) <= 8 
            subplot(3, 3, find(strcmp(p,plots))) ;
        end
        
        plot(1:size(values,2), values(:, 1:end)','.-') ; % don't plot the first epoch
        xlabel('epoch') ;
        
        if isempty(strfind(lower(p), 'acc'))
            [minVal,minIdx] = min(values(2,:));
            [minValTr,minIdxTr] = min(values(1,:));
            title(sprintf('%s ts%.4f (%d) tr%.4f (%d) ', p, min(values(2,:)), minIdx, min(values(1,:)),minIdxTr), 'Interpreter', 'none') ;
            %         title(sprintf('%s tsErr%.4f (%d) trErr%.4f', p, min(values(2,:)), minIdx, min(values(1,:)))) ;
            legend(leg{:},'location', 'SouthOutside') ;
            grid on ;
        else
            [maxVal,maxIdx] = max(values(2,:));
            [maxValTr,maxIdxTr] = max(values(1,:));
            title(sprintf('%s ts%.4f (%d) tr%.4f (%d) ', p, max(values(2,:)), maxIdx, max(values(1,:)), maxIdxTr), 'Interpreter', 'none') ;
            legend(leg{:},'location', 'SouthOutside') ;
            grid on ;
        end
    end
    drawnow ;
end