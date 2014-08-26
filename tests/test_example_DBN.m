function er = test_example_DBN(ifAsy, thread_num, fetch, push)
load mnist_uint8;

disp(['ifAsy=', num2str(double(ifAsy)) , ', thread_num=', num2str(thread_num), ' n_fetch=', num2str(fetch), ', n_push=', num2str(push), '.']);
if ~ifAsy && thread_num~=1
    p = thread_num;
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
% opts.thread_num = thread_num;
% opts.n_fetch = fetch;
% opts.n_push = push;
% dbn = dbnsetup(dbn, train_x, opts);
% dbn = dbntrain(dbn, train_x, opts);
% figure; visualize(dbn.rbm{1}.W');   %  Visualize the RBM weights

%%  ex2 train a 100-100 hidden unit DBN and use its weights to initialize a NN
% rand('state',0)
%train dbn
dbn.sizes = [100 100];
opts.numepochs =   10;
opts.batchsize = 100;
opts.momentum  =   0;
opts.alpha     =   1;
dbn = dbnsetup(dbn, train_x, opts);
tic;
dbn = dbntrain(dbn, train_x, opts);
toc;
% figure; visualize(dbn.rbm{1}.W');   %  Visualize the RBM weights

%unfold dbn to nn
nn = dbnunfoldtonn(dbn, 10);
nn.activation_function = 'sigm';

%train nn
opts.numepochs =  10;
opts.batchsize = 100;
nn = nntrain(nn, train_x, train_y, opts);
[er, bad] = nntest(nn, test_x, test_y);
disp([num2str(er*100), '%']);

% assert(er < 0.10, 'Too big error');
