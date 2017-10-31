function saveState(fileName, net, stats)
% -------------------------------------------------------------------------
net_ = net ;
net = net_.saveobj() ;
save(fileName, 'net', 'stats') ;
