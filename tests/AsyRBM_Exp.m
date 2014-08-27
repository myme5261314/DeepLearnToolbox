diary('log.txt');
diary on;
disp('v0.06');

% error_list = zeros(20, 10);
asy_error_list = zeros(20, 10);

n_fetch = 10;
n_push = 20;

tic;

%% Normal RBM split dataset performance.
% for i=1:size(error_list,1)
%     for j=1:size(error_list,2)
%         error_list(i,j) = test_example_DBN(false, i, 1, 1);
%     end
% end

%% Asy RBM thread_num performance.
for i=1:size(asy_error_list,1)
    for j=1:size(asy_error_list,2)
        asy_error_list(i,j) = test_example_DBN(true, i, n_fetch, n_push);
    end
end

toc;

save('metrics.mat', 'asy_error_list');

% rbm_accuracy = 1-mean(error_list,2);
asy_rbm_accuracy = 1-mean(asy_error_list,2);

figure;
% plot(1:size(error_list,1), rbm_accuracy, 'red');
% hold all;
plot(1:size(asy_error_list,1), asy_rbm_accuracy, 'blue');

diary off;