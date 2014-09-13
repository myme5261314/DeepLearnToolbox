function dbn = dbntrain(dbn, x, opts)
    n = numel(dbn.rbm);

    dbn.rbm{1} = rbmtrain(dbn.rbm{1}, x, opts);
    for i = 2 : n
        x = rbmup(dbn.rbm{i - 1}, x);
        dbn.rbm{i} = rbmtrain(dbn.rbm{i}, x, opts);
    end
    
    if opts.ifdropout
        for i=1:n
            dbn.rbm{i}.W = dbn.rbm{i}.W / opts.dropout;
        end
    end

end
