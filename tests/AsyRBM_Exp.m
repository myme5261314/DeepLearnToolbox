diary('log.txt');
diary on;
disp('v0.13');

error_list = zeros(20, 10);
% asy_error_list = zeros(20, 10);
asy_error_list_dropout = zeros(20, 10);
opts.numepochs =   30;
opts.batchsize = 100;
opts.momentum  =   0;
opts.alpha     =   1;
opts.n_fetch = 20;
opts.n_push = 10;
opts.weightPenaltyL2 = 0;
%% For ARBM
opts.ifAsy = 0;
%% For ARBM Dropout
opts.ifdropout = 1;
opts.dropout = 0.5;

opts.sizes = [100 100];
% opts.sizes = [500 500 2000];

tic;

%% Normal RBM split dataset performance.
opts.ifAsy = 0;
opts.ifdropout = 0;
if ~exist('normal-rbm.mat', 'file')
    for i=1:size(error_list,1)
        for j=1:size(error_list,2)
            disp(['Normal RBM- thread_num=', num2str(i), ', exp_index=', num2str(j)]);
            tic;
            opts.thread_num = i;
            error_list(i,j) = test_example_DBN(opts);
            toc;
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
opts.ifdropout = 0;
n_push_fetch = [5 5; 5 10; 5 15; 5 20; 10 20; 15 20; 20 20];
asy_error_list = cell( size(n_push_fetch,1), 1);
for i=1:size(n_push_fetch,1)
    asy_error_list{i} = zeros(20, 10);
end
if ~exist('asy-rbm.mat', 'file')
    for k=1:size(asy_error_list,1)
        opts.n_push = n_push_fetch(k,1);
        opts.n_fetch = n_push_fetch(k,2);
        for i=2:size(asy_error_list{k},1)
            for j=1:size(asy_error_list{k},2)
                disp(['Asy RBM- thread_num=', num2str(i), ', exp_index=', num2str(j)]);
                tic;
                opts.thread_num = i;
                asy_error_list{k}(i,j) = test_example_DBN(opts);
                toc;
            end
        end
    end
    save('asy-rbm.mat', 'asy_error_list');
else
    load('asy-rbm.mat');
end

% for i=1:size(asy_error_list,1)
%     plot(1:size(asy_error_list,2), asy_error_list(i,:), 'blue');
%     hold all;
% end
% 
% figure('Name', 'Asy RBM');
% for i=1:size(asy_error_list,1)
%     plot(1:size(asy_error_list,2), asy_error_list(i,:), 'blue');
%     hold all;
% end

%% Asy with dropout RBM thread_num performance.
opts.ifdropout = 1;
if ~exist('asy-rbm-dropout.mat', 'file')
    for i=1:size(asy_error_list_dropout,1)
        for j=1:size(asy_error_list_dropout,2)
            disp(['Asy RBM with dropout- thread_num=', num2str(i), ', exp_index=', num2str(j)]);
            tic;
            opts.thread_num = i;
            asy_error_list_dropout(i,j) = test_example_DBN(opts);
            toc;
        end
    end
    save('asy-rbm-dropout.mat', 'asy_error_list_dropout');
else
    load('asy-rbm-dropout.mat');
end

toc;

% save('metrics.mat', 'asy_error_list');

%% Calculate mean accuracy

% rbm_accuracy = 1-mean(error_list,2);
% asy_rbm_accuracy = 1-mean(asy_error_list,2);
% asy_rbm_dropout_accuracy = 1-mean(asy_error_list_dropout,2);


%% plot
% figure('Name', 'Normal vs Asy RBM mean');
% plot(1:size(error_list,1), rbm_accuracy, 'r--o', 'MarkerSize', 3);
% hold all;
% plot(1:size(asy_error_list,1), asy_rbm_accuracy(1:end), 'b--s', 'MarkerSize', 3);
% hold all;
% plot(1:size(asy_error_list_dropout,1), asy_rbm_dropout_accuracy(1:end), 'g--d', 'MarkerSize', 3);
% legend('Normal', 'S2-ARBM', 'S2-ARBM-Dropout');

diary off;