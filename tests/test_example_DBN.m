function er = test_example_DBN(opts)
load data_batch_1;
train_x = data;
labelset = labels;
load data_batch_2;
train_x = [train_x; data];
labelset = [labelset; labels];
load data_batch_3;
train_x = [train_x; data];
labelset = [labelset; labels];
load data_batch_4;
train_x = [train_x; data];
labelset = [labelset; labels];
load data_batch_5;
train_x = [train_x; data];
labelset = [labelset; labels];
train_y = zeros(size(labelset,1),10);
train_y(sub2ind(size(train_y), 1:size(labelset,1), labelset'+1))=1;
for i=1:size(train_x,1)
    train_x(i,:) = reshape(permute(reshape(train_x(i,:), [32 32 3]), [2 1 3]), [1 32*32*3]);
end
load test_batch;
test_x = data;
test_y = zeros(size(labels,1),10);
test_y(sub2ind(size(test_y), 1:size(labels,1), labels'+1))=1;
for i=1:size(test_x,1)
    test_x(i,:) = reshape(permute(reshape(test_x(i,:), [32 32 3]), [2 1 3]), [1 32*32*3]);
end


% disp(['ifAsy=', num2str(double(opts.ifAsy)) , ', thread_num=', num2str(opts.thread_num), ' n_fetch=', num2str(opts.n_fetch), ', n_push=', num2str(opts.n_push), '.']);
disp(opts);
if ~opts.ifAsy
    p = opts.thread_num;
    ridx = randperm(size(train_x,1));
    train_x = train_x(ridx(1:floor(size(ridx,2)/p)), :);
    train_y = train_y(ridx(1:floor(size(ridx,2)/p)), :);
end


train_x = double(train_x) / 255;
test_x  = double(test_x)  / 255;
train_y = double(train_y);
test_y  = double(test_y);

%%  ex1 train a 100 hidden unit RBM and visualize its weights
% rand('state',0)
% dbn.sizes = [100];
% opts.numepochs =   1;
% opts.batchsize = 100;
% opts.momentum  =   0;
% opts.alpha     =   1;

% dbn = dbnsetup(dbn, train_x, opts);
% dbn = dbntrain(dbn, train_x, opts);
% figure; visualize(dbn.rbm{1}.W');   %  Visualize the RBM weights

%%  ex2 train a 100-100 hidden unit DBN and use its weights to initialize a NN
% rand('state',0)
%train dbn
dbn.sizes = opts.sizes;
% opts.numepochs =   30;
% opts.batchsize = 100;
% opts.momentum  =   0.9;
% opts.alpha     =   0.1;

tic;
dbn = dbnsetup(dbn, train_x, opts);
dbn = dbntrain(dbn, train_x, opts);
toc;
% figure; visualize(dbn.rbm{1}.W');   %  Visualize the RBM weights

%unfold dbn to nn
nn = dbnunfoldtonn(dbn, 10);
nn.activation_function = 'sigm';
nn.weightPenaltyL2 = opts.weightPenaltyL2;

%train nn
%opts.numepochs =  100;
opts.alpha = 0.1;
% opts.batchsize = 100;
tic;
nn = nntrain(nn, train_x, train_y, opts);
[er, bad] = nntest(nn, test_x, test_y);
[er1, bad1] = nntest(nn, train_x, train_y);
toc;
disp([num2str(er*100), '%']);

% assert(er < 0.10, 'Too big error');
