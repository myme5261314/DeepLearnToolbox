function X = sigmrnd(P)
%     X = double(1./(1+exp(-P)))+1*randn(size(P));
    X = single(1./(1+exp(-P)) > gpuArray(single(rand(size(P)))));
end