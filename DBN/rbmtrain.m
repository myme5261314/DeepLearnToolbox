function rbm = rbmtrain(rbm, x, opts)
    assert(isfloat(x), 'x must be a float');
    assert(all(x(:)>=0) && all(x(:)<=1), 'all data in x must be in [0:1]');
%     x = [x; x];
%     ridx = randperm(size(x,1));
%     x = x(ridx(1:size(x,1)/60), :);
    m = size(x, 1);
    numbatches = floor(m / opts.batchsize);
    
    assert(rem(numbatches, 1) == 0, 'numbatches not integer');
    
    if ~opts.ifAsy
        opts.thread_num = 1;
        opts.n_fetch = 1;
        opts.n_push = 1;
    end
    thread_num = opts.thread_num;
    n_fetch = opts.n_fetch;
    n_push = opts.n_push;
    W = cell(thread_num, 1);
    vW = cell(thread_num, 1);
    accuredW = cell(thread_num, 1);
    for i = 1 : thread_num
        W{i} = zeros(size(rbm.W));
        vW{i} = zeros(size(rbm.W));
        accuredW{i} = zeros(size(rbm.W));
    end
    b = cell(thread_num, 1);
    vb = cell(thread_num, 1);
    accuredb = cell(thread_num, 1);
    for i = 1 : thread_num
        b{i} = zeros(size(rbm.b));
        vb{i} = zeros(size(rbm.b));
        accuredb{i} = zeros(size(rbm.b));
    end
    c = cell(thread_num, 1);
    vc = cell(thread_num, 1);
    accuredc = cell(thread_num, 1);
    for i = 1 : thread_num
        c{i} = zeros(size(rbm.c));
        vc{i} = zeros(size(rbm.c));
        accuredc{i} = zeros(size(rbm.c));
    end
    for i = 1 : opts.numepochs
        kk = randperm(m);
        err = 0;
        thread_index = randi(thread_num, numbatches);
        index_step = zeros(thread_num, 1);
        for l = 1 : numbatches
            batch = x(kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize), :);
            one_index = thread_index(l);
            if mod( index_step(one_index), n_fetch ) == 0
                W{one_index} = rbm.W;
                b{one_index} = rbm.b;
                c{one_index} = rbm.c;
            end
            if opts.ifdropout
                dropOut = rand(opts.batchsize, size(rbm.W, 1)) > opts.dropout;
            end
            v1 = batch;
            h1 = sigmrnd(repmat(c{one_index}', opts.batchsize, 1) + v1 * W{one_index}');
            if opts.ifdropout
                h1 = h1.*dropOut;
            end
            v2 = sigmrnd(repmat(b{one_index}', opts.batchsize, 1) + h1 * W{one_index});
            h2 = sigm(repmat(c{one_index}', opts.batchsize, 1) + v2 * W{one_index}');
            if opts.ifdropout
                h2 = h2.*dropOut;
            end

            c1 = h1' * v1;
            c2 = h2' * v2;

            vW{one_index} = rbm.momentum * vW{one_index} + rbm.alpha * (c1 - c2)     / opts.batchsize;
            vb{one_index} = rbm.momentum * vb{one_index} + rbm.alpha * sum(v1 - v2)' / opts.batchsize;
            vc{one_index} = rbm.momentum * vc{one_index} + rbm.alpha * sum(h1 - h2)' / opts.batchsize;

            accuredW{one_index} = accuredW{one_index} + vW{one_index};
            accuredb{one_index} = accuredb{one_index} + vb{one_index};
            accuredc{one_index} = accuredc{one_index} + vc{one_index};
            W{one_index} = W{one_index} + vW{one_index};
            b{one_index} = b{one_index} + vb{one_index};
            c{one_index} = c{one_index} + vc{one_index};

            err = err + sum(sum((v1 - v2) .^ 2)) / opts.batchsize;
            
            if mod( index_step(one_index), n_push )  == 0
% %                 rbm.W = rbm.W + accuredW{one_index};
% %                 rbm.b = rbm.b + accuredb{one_index};
% %                 rbm.c = rbm.c + accuredc{one_index};
%                 w1 = 1/thread_num;
% %                 w1 = 1/sqrt(thread_num);
% %                 w1 = 1-1/thread_num;
% %                 w1 = 0;
%                 w2 = 1-w1;
%                 if thread_num == 1
%                     w1 = 0;
%                     w2 = 1;
%                 end
%                 rbm.W = w1*rbm.W + w2*W{one_index};
%                 rbm.b = w1*rbm.b + w2*b{one_index};
%                 rbm.c = w1*rbm.c + w2*c{one_index};
                rbm.W = rbm.W + accuredW{one_index};
                rbm.b = rbm.b + accuredb{one_index};
                rbm.c = rbm.c + accuredc{one_index};
                accuredW{one_index} = zeros(size(rbm.W));
                accuredb{one_index} = zeros(size(rbm.b));
                accuredc{one_index} = zeros(size(rbm.c));
            end
            
            index_step(one_index) = index_step(one_index) + 1;
            
        end
        
        disp(['epoch ' num2str(i) '/' num2str(opts.numepochs)  '. Average reconstruction error is: ' num2str(err / numbatches)]);
        
    end
end
