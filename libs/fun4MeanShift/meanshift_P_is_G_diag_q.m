classdef meanshift_P_is_G_diag_q < dagnn.ElementWise
    properties        
        delta = 0.1
    end
    properties (Transient)
        numInputs
        SIZE_
    end
    
    % [TODO]: current version only supports batchSize=1; need to extend to 
    % multiple input images    
    methods        
        function outputs = forward(obj, inputs, params)
            obj.numInputs = numel(inputs); 
            G = inputs{1}; 
            q = inputs{2};
            %obj.SIZE_ = inputs{3};
            gpuMode = isa(G, 'gpuArray');
            % [height width channels batchsize]
            if gpuMode
                P = gpuArray(zeros(size(G,1), size(G,2), 1, size(G,4), 'single'));
            else
                P = zeros(size(G,1), size(G,2), 1, size(G,4), 'single');
            end
            for i = 1:size(G,4)
                P(:,:,:,i) = G(:,:,:,i)*diag(q(:,:,:,i));
            end
            outputs{1} = P;
        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            derInputs = cell(1, numel(inputs));
                        
            dzdy = derOutputs{1};           
            G = inputs{1}; 
            q = inputs{2};
            gpuMode = isa(G, 'gpuArray');
            
            if gpuMode
                dzdx_G = gpuArray(zeros(size(G,1), size(G,2), size(G,3), size(G,4), 'single'));
                dzdx_q = gpuArray(zeros(size(q,1), size(q,2), size(q,3), size(q,4), 'single'));
            else
                dzdx_G = zeros(size(G,1), size(G,2), size(G,3), size(G,4), 'single');
                dzdx_q = zeros(size(q,1), size(q,2), size(q,3), size(q,4), 'single');
            end
            for i = 1:size(G,4)
                cur_G = G(:,:,:,i);
                cur_q = q(:,:,:,i);
                cur_dzdx_G = dzdy(:,:,:,i) * diag(cur_q);
%                 cur_dzdx_q = dzdy(:,:,:,i)'*sum(cur_G,2);
                
                cur_dzdx_q = dzdy(:,:,:,i).*cur_G;
                cur_dzdx_q = sum(cur_dzdx_q,1);
                
                dzdx_G(:,:,:,i) = cur_dzdx_G;
                dzdx_q(:,:,:,i) = cur_dzdx_q';
            end
%             
%             dzdx_G = dzdy*repmat(q,1,size(G,2));
%             dzdx_q = dzdy'*sum(G,2)';
            derInputs{1} = dzdx_G;
            derInputs{2} = dzdx_q;
            derParams = {} ;            
        end
        
        function obj = meanshift_P_is_G_diag_q(varargin)
            obj.load(varargin) ;
        end
    end
end
