diary('log.txt');
diary on;

disp('v0.05');
error_list = zeros(20, 10);
% asy_error_list = zeros(20, 10);

%% Normal RBM split dataset performance.
for i=1:size(error_list,1)
    for j=1:size(error_list,2)
        error_list(i,j) = test_example_DBN(false, i, 1, 1);
    end
end

% %% Asy RBM thread_num performance.
% for i=1:size(asy_error_list,1)
%     for j=1:size(asy_error_list,2)
%         asy_error_list(i,j) = test_example_DBN(true, i, 10, 5);
%     end
% end

rbm_accuracy = 1-mean(error_list,2);
% asy_rbm_accuracy = 1-mean(asy_error_list);

figure;
plot(1:size(error_list,1), rbm_accuracy, 'red');
% hold all;
% plot(1:size(asy_error_list,1), asy_rbm_accuracy, 'blue');
save('metrics.mat', 'error_list');

diary off;