diary('log.txt');
diary on;
disp('v0.11');

error_list = zeros(20, 10);
asy_error_list = zeros(20, 10);
asy_error_list_dropout = zeros(20, 10);
opts.numepochs =   30;
opts.batchsize = 100;
opts.momentum  =   0;
opts.alpha     =   1;
opts.n_fetch = 10;
opts.n_push = 20;
opts.weightPenaltyL2 = 0;
%% For ARBM
opts.ifAsy = 0;
opts.thread_num = 12;
opts.n_fetch = 10;
opts.n_push = 20;
%% For ARBM Dropout
opts.ifdropout = 1;
opts.dropout = 0.5;

opts.sizes = [100 100];
% opts.sizes = [500 500 2000];

tic;

%% Normal RBM split dataset performance.
opts.ifAsy = 0;
opts.n_fetch = 1;
opts.n_push = 1;
opts.ifdropout = 0;
if ~exist('normal-rbm.mat', 'file')
    for i=1:size(error_list,1)
        for j=1:size(error_list,2)
            opts.thread_num = i;
            error_list(i,j) = test_example_DBN(opts);
        end
    end
    save('normal-rbm.mat', 'error_list');
else
    load('normal-rbm.mat');
end
figure('Name', 'Normal RBM');
for i=1:size(error_list,1)
    plot(1:size(error_list,2), error_list(i,:), 'red');
    hold all;
end
figure('Name', 'Normal vs Asy RBM');
for i=1:size(error_list,1)
    plot(1:size(error_list,2), error_list(i,:), 'red');
    hold all;
end

%% Asy without Dropout RBM thread_num performance.
opts.ifAsy = 1;
opts.n_fetch = 10;
opts.n_push = 20;
opts.ifdropout = 0;
if ~exist('asy-rbm.mat', 'file')
    for i=1:size(asy_error_list,1)
        for j=1:size(asy_error_list,2)
            opts.thread_num = i;
            asy_error_list(i,j) = test_example_DBN(opts);
        end
    end
    save('asy-rbm.mat', 'asy_error_list');
else
    load('asy-rbm.mat');
end

for i=1:size(asy_error_list,1)
    plot(1:size(asy_error_list,2), asy_error_list(i,:), 'blue');
    hold all;
end

figure('Name', 'Asy RBM');
for i=1:size(asy_error_list,1)
    plot(1:size(asy_error_list,2), asy_error_list(i,:), 'blue');
    hold all;
end

%% Asy with dropout RBM thread_num performance.
opts.ifdropout = 1;
if ~exist('asy-rbm-dropout.mat', 'file')
    for i=1:size(asy_error_list_dropout,1)
        for j=1:size(asy_error_list_dropout,2)
            opts.thread_num = i;
            asy_error_list_dropout(i,j) = test_example_DBN(opts);
        end
    end
    save('asy-rbm-dropout.mat', 'asy_error_list_dropout');
else
    load('asy-rbm-dropout.mat');
end

toc;

% save('metrics.mat', 'asy_error_list');

rbm_accuracy = 1-mean(error_list,2);
asy_rbm_accuracy = 1-mean(asy_error_list,2);
asy_rbm_dropout_accuracy = 1-mean(asy_error_list_dropout,2);


figure('Name', 'Normal vs Asy RBM mean');
plot(1:size(error_list,1), rbm_accuracy, 'r--o', 'MarkerSize', 3);
hold all;
plot(1:size(asy_error_list,1), asy_rbm_accuracy(1:end), 'b--s', 'MarkerSize', 3);
hold all;
plot(1:size(asy_error_list_dropout,1), asy_rbm_dropout_accuracy(1:end), 'g--d', 'MarkerSize', 3);
legend('Normal', 'S2-ARBM', 'S2-ARBM-Dropout');

diary off;