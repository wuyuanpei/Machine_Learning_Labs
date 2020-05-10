function [ w, e_in ] = logistic_reg( X, y, w_init, max_its, eta )
%LOGISTIC_REG Learn logistic regression model using gradient descent
%   Inputs:
%       X : data matrix (without an initial column of 1s)
%       y : data labels (plus or minus 1)
%       w_init: initial value of the w vector (d+1 dimensional)
%       max_its: maximum number of iterations to run for
%       eta: learning rate
    
%   Outputs:
%       w : weight vector
%       e_in : in-sample error (the cross-entropy error as defined in LFD)
    [N, ~] = size(X);
    X = [ones(N,1) X]; % add an inital column of 1
    w = w_init;

    G_numerator = X.*y;
    for i = 1:max_its
        G_denominator = X*w;
        G_denominator = G_denominator.*y;
        G_denominator = 1 + exp(G_denominator);
        G = G_numerator./G_denominator;
        G = (mean(G))';
        w = w + eta*G;
        % terminate condition
        if (1e-3)-abs(G) > 0
            break;
        end
    end

    e_in = mean(log(exp(-((X*w).*y))+1));
   
end


% function [ w, e_in ] = logistic_reg( X, y, w_init, max_its, eta )
% %LOGISTIC_REG Learn logistic regression model using stochastic GD
% %   Inputs:
% %       X : data matrix (without an initial column of 1s)
% %       y : data labels (plus or minus 1)
% %       w_init: initial value of the w vector (d+1 dimensional)
% %       max_its: maximum number of iterations to run for
% %       eta: learning rate
%     
% %   Outputs:
% %       w : weight vector
% %       e_in : in-sample error (the cross-entropy error as defined in LFD)
%     [N, ~] = size(X);
%     X = [ones(N,1) X]; % add an inital column of 1
%     w = w_init;
% 
%     
%     for i = 1:max_its
%         n = randi(N);
%         G_numerator = y(n)*X(n,:);
%         G_denominator = X(n,:)*w;
%         G_denominator = G_denominator*y(n);
%         G_denominator = 1 + exp(G_denominator);
%         G = G_numerator/G_denominator;
%         w = w + eta*G';
%         % terminate condition
%         if (1e-3)-abs(G) > 0
%             break;
%         end
%     end
% 
%     e_in = mean(log(exp(-((X*w).*y))+1));
%    
% end
