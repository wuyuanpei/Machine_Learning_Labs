%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all;
T = csvread('clevelandtrain.csv', 1, 0);
y = T(:,14)*2-1;
[X,MU,SIGMA] = zscore(T(:,1:13));
w_init = zeros(14,1);
max_its = 1e6;
eta = 0.01;
tic
[w, e_in] = logistic_reg_regularized(X, y, w_init, max_its, eta, 0.1);    
toc
test_error_1 = find_test_error(w, X, y);

T2 = csvread('clevelandtest.csv', 1, 0);
y2 = T2(:,14)*2-1;
X2 = (T2(:,1:13)-MU)./SIGMA;
test_error_2 = find_test_error(w, X2, y2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% clear all;
% T = csvread('clevelandtrain.csv', 1, 0);
% y = T(:,14);
% X = T(:,1:13);
% 
% tic
% w = glmfit(X, y, 'binomial');
% toc
% 
% y = T(:,14)*2-1;
% [N, ~] = size(X);
% X = [ones(N,1) X]; % add an inital column of 1
% e_in = mean(log(exp(-((X*w).*y))+1));
% X = T(:,1:13);
% test_error_1 = find_test_error(w, X, y);
%  
% T2 = csvread('clevelandtest.csv', 1, 0);
% y2 = T2(:,14)*2-1;
% X2 = T2(:,1:13);
% test_error_2 = find_test_error(w, X2, y2);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% clear all;
% T = csvread('clevelandtrain.csv', 1, 0);
% y = T(:,14)*2-1;
% [X,MU,SIGMA] = zscore(T(:,1:13));
% w_init = zeros(14,1);
% eta = 0.01;
% 
% tic
% [N, d] = size(X);
% X = [ones(N,1) X]; % add an inital column of 1
% w = w_init;
% 
% G_numerator = X.*y;
% 
% iter = 0;
% while 1==1
%     iter = iter + 1;
%     G_denominator = X*w;
%     G_denominator = G_denominator.*y;
%     G_denominator = 1 + exp(G_denominator);
%     G = G_numerator./G_denominator;
%     G = (mean(G))';
%     w = w + eta*G;
%     % terminate condition
%     if (1e-6)-abs(G) > 0
%         break;
%     end
% end
% 
% toc
% 
% e_in = mean(log(exp(-((X*w).*y))+1));
% 
% T2 = csvread('clevelandtest.csv', 1, 0);
% y2 = T2(:,14)*2-1;
% X2 = (T2(:,1:13)-MU)./SIGMA;
% test_error_2 = find_test_error(w, X2, y2);
