function rbm = rbmtrain(rbm, x, opts)
    assert(isfloat(x), 'x must be a float');
%     assert(all(x(:)>=0) && all(x(:)<=1), 'all data in x must be in [0:1]');
    m = size(x, 1);
    numbatches = m / opts.batchsize;
    
    assert(rem(numbatches, 1) == 0, 'numbatches not integer');
    
    g_W = gpuArray(single(rbm.W));
    g_vW = gpuArray(single(rbm.vW));
    g_c = gpuArray(single(rbm.c));
    g_vc = gpuArray(single(rbm.vc));
    g_b = gpuArray(single(rbm.b));
    g_vb = gpuArray(single(rbm.vb));
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
            g_v1 = gpuArray(single(batch));
            g_h1 = sigmrnd(bsxfun(@plus, g_c', g_v1*g_W'));
%             g_v2 = sigmrnd(bsxfun(@plus, g_b', g_h1*g_W));
            g_v2 = normrnd(bsxfun(@plus, g_b', g_h1*g_W), 1);
            g_h2 = sigm(bsxfun(@plus, g_c', g_v2*g_W'));

            g_c1 = g_h1' * g_v1;
            g_c2 = g_h2' * g_v2;

            g_vW = rbm.momentum * g_vW + rbm.alpha * ((g_c1 - g_c2)     / opts.batchsize - opts.L2*g_W);
            g_vb = rbm.momentum * g_vb + rbm.alpha * sum(g_v1 - g_v2)' / opts.batchsize;
            g_vc = rbm.momentum * g_vc + rbm.alpha * sum(g_h1 - g_h2)' / opts.batchsize;

            g_W = g_W + g_vW;
            g_b = g_b + g_vb;
            g_c = g_c + g_vc;

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
