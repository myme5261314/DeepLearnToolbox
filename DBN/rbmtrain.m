function rbm = rbmtrain(rbm, x, opts)
%     assert(isfloat(x), 'x must be a float');
%     assert(all(x(:)>=0) && all(x(:)<=1), 'all data in x must be in [0:1]');
    m = size(x, 1);
    numbatches = m / opts.batchsize;
    
    assert(rem(numbatches, 1) == 0, 'numbatches not integer');
    
    idx = [1:30 121:150 241:270];
    g_W = gpuArray(single(rbm.W));
    g_vW = gpuArray(single(rbm.vW));
    g_c = gpuArray(single(rbm.c));
    g_vc = gpuArray(single(rbm.vc));
    g_b = gpuArray(single(rbm.b));
%     g_b(idx) = 0;
    g_vb = gpuArray(single(rbm.vb));
%     g_vb(idx) = 0;
    fixExpRnd = @(mumat) -mumat .* log(rand(size(mumat)));
    for i = 1 : opts.numepochs
        tic;
        kk = randperm(m);
        err = 0;
        for l = 1 : numbatches
            batch = x(kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize), :);
%             v1 = batch;
%             h1 = sigmrnd(repmat(rbm.c', opts.batchsize, 1) + v1 * rbm.W');
%             v2 = sigmrnd(repmat(rbm.b', opts.batchsize, 1) + h1 * rbm.W);
%             h2 = sigm(repmat(rbm.c', opts.batchsize, 1) + v2 * rbm.W');
            batch = full(batch);
            batch = single(batch);
            if isfield(opts, 'mu') && isfield(opts, 'sigma')
                sigma_idx = find(opts.sigma);
                batch = bsxfun(@minus, batch, opts.mu);
                batch(:,sigma_idx) = bsxfun(@minus, batch(:,sigma_idx), opts.sigma(:,sigma_idx));
            end
            g_v1 = gpuArray(single(batch));
            g_h1 = sigmrnd(bsxfun(@plus, g_c', g_v1*g_W'));
%             g_v2 = sigmrnd(bsxfun(@plus, g_b', g_h1*g_W));
            g_v2 = gpuArray(zeros(size(g_v1),'single'));
            
            temp = bsxfun(@plus, g_b', g_h1*g_W);
            temp1 = sigmrnd(temp(:,setdiff(1:360,idx)));
            g_v2(:,setdiff(1:360,idx)) = temp1;
            temp1 = temp(:,idx);
            temp1 = round(fixExpRnd(temp1));
%             temp1 = exprnd(temp1);
%             temp1 = abs(normrnd(temp1,1));
%             temp = temp.^2;
%             temp1 = exprnd(1./temp);
%             temp1 = temp(:,idx);
%             if nnz(temp1<0)~=0
%                 fprintf('fix %d g_v2 values to 0.01 .\n', nnz(temp1<0));
%                 temp1(temp1<0) = 0.01;
%                 temp(:,idx) = temp1;
%             end
%             clear temp1;
%             temp1 = temp(:,idx);
%             tempidx = find(temp1~=0);
%             temp1(tempidx) = 1./temp(tempidx);
%             temp1 = exprnd(temp1);
            assert(gather(any(any(isnan(temp1)))==0));
%             if (gather(any(any(isnan(temp1))))==1)
%                 disp(temp1);
%                 pause;
%             end
            g_v2(:,idx) = temp1;
%             g_v2 = temp1;
%             idx = [31:120 151:240 271:360];
%             g_v2(:,setdiff(1:360,idx)) = sigmrnd(temp(:,setdiff(1:360,idx)));
%             clear temp;
%             g_v2 = normrnd(bsxfun(@plus, g_b', g_h1*g_W), 1);
%             g_v2 = bsxfun(@plus, g_b', g_h1*g_W);% + gpuArray(randn(size(g_v1), 'single'));
% 			g_v2 = normrnd(bsxfun(@rdivide, bsxfun(@plus, g_b', g_h1*g_W), 2), sqrt(0.5));
            g_h2 = sigm(bsxfun(@plus, g_c', g_v2*g_W'));

            g_c1 = g_h1' * g_v1;
            g_c2 = g_h2' * g_v2;

            g_vW = rbm.momentum * g_vW + rbm.alpha * ((g_c1 - g_c2)     / opts.batchsize - opts.L2*g_W);
            g_vb = rbm.momentum * g_vb + rbm.alpha * sum(g_v1 - g_v2)' / opts.batchsize;
%             g_vb(idx) = 0;
%             g_vc = rbm.momentum * g_vc + rbm.alpha * sum(g_h1 - g_h2)' / opts.batchsize;

            g_W = g_W + g_vW;
            g_b = g_b + g_vb;
%             g_c = g_c + g_vc;
%             temp = g_W(:,idx);
%             if nnz(temp<0)~=0
%                 fprintf('fix %d g_W,g_vW values to 0.\n', nnz(temp<0));
%                 temp(temp<0) = 0;
%                 g_W(:,idx) = temp;
% %                 temp = g_b(idx);
% %                 temp(temp<0) = 0;
% %                 g_b(idx) = temp;
%                 temp = g_vW(:,idx);
%                 temp(temp<0) = 0;
%                 g_vW(:,idx) = temp;
%                 
%             end

            err = err + gather(sum(sum((g_v1 - g_v2) .^ 2))) / opts.batchsize;
            
        end
        toc;
        disp(['epoch ' num2str(i) '/' num2str(opts.numepochs)  '. Average reconstruction error is: ' num2str(err / numbatches)]);
        
    end
    rbm.W = gather(g_W);
    rbm.vW = gather(g_vW);
    rbm.c = gather(g_c);
    rbm.vc = gather(g_vc);
    rbm.b = gather(g_b);
    rbm.vb = gather(g_vb);
end
