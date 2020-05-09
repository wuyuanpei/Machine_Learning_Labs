clear all
[num_iters, bounds] = perceptron_experiment(100,10,1000);

hist(num_iters,20);
xlabel('Number of iterations')
ylabel('Times')
title('Number of iterations PLA takes to learn a linear separator')
diff = bounds-num_iters;
figure();
hist(log(diff),20);
xlabel('log(bound - the number of iterations)')
ylabel('Times')
title('Difference between the bound and the number of iterations')

% N = 100;
% d = 10;
% %%%%%%%%%%%%%% perceptron_experiment()
% w_optimal = [0, rand(1, d)];
% w_optimal_norm = norm(w_optimal);%%%
% D = ones(N, 1);
% D(:,2:d + 1) = 2 * rand(N, d) - 1;
% x_norm = sum(abs(D).^2,2).^(1/2);
% R = max(x_norm);%%%
% 
% Y = sign(D * w_optimal');
% Rho = min(abs(D * w_optimal'));
% D(:,d+2) = Y;
% %%%%%%%%%%%%%% Pass D to perceptron_learn()
% data_in = D;
% %%%%%%%%%%%%%% perceptron_learn()
% data_size = size(data_in);
% w_length = data_size(2) - 1;
% w = zeros(1, w_length);
% data_x = data_in(:,1:w_length);
% data_y = data_in(:,data_size(2));
% iterations = 0;
% while 1 == 1
%     iterations = iterations + 1;
%     y_classified = sign(data_x * w');
%     y_difference = y_classified - data_y;
%     different_idx = find(y_difference);
%     if isempty(different_idx)
%         break;
%     else
%         idx = different_idx(1);
%         w = w + data_y(idx)*data_x(idx,:);
%     end
% end
% % % %%%%%%%%%%%%%%% verify
% % %a = sign(data_x*w')
% % %b = sign(data_x*w_optimal')