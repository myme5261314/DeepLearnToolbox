function dbn = dbnsetup(dbn, x, opts)
    n = size(x, 2);
    dbn.sizes = [n, dbn.sizes];

    for u = 1 : numel(dbn.sizes) - 1
        dbn.rbm{u}.alpha    = opts.alpha;
        dbn.rbm{u}.momentum = opts.momentum;

        dbn.rbm{u}.W  = ones(dbn.sizes(u + 1), dbn.sizes(u));
        dbn.rbm{u}.vW = zeros(dbn.sizes(u + 1), dbn.sizes(u));
        dbn.rbm{u}.W = (rand(dbn.sizes(u + 1), dbn.sizes(u)))*8*sqrt(6/(dbn.sizes(u + 1)+dbn.sizes(u)));
%         dbn.rbm{u}.W = 0.01*randn(dbn.sizes(u + 1), dbn.sizes(u));
%         dbn.rbm{u}.vW = dbn.rbm{u}.W;
        

        dbn.rbm{u}.b  = zeros(dbn.sizes(u), 1);
        dbn.rbm{u}.vb = zeros(dbn.sizes(u), 1);
%         dbn.rbm{u}.b  = 4*ones(dbn.sizes(u), 1);
%         dbn.rbm{u}.vb = dbn.rbm{u}.b;

%         dbn.rbm{u}.c  = zeros(dbn.sizes(u + 1), 1);
        dbn.rbm{u}.vc = zeros(dbn.sizes(u + 1), 1);
        dbn.rbm{u}.c = -10*ones(dbn.sizes(u+1), 1);
%         dbn.rbm{u}.vc = dbn.rbm{u}.c;
    end

end
