function [ test_error ] = find_test_error( w, X, y )
% find_test_error: compute the test error of a linear classifier w. The
%  hypothesis is assumed to be of the form sign([1 x(n,:)] * w)
%  Inputs:
%		w: weight vector
%       X: data matrix (without an initial column of 1s)
%       y: data labels (plus or minus 1)
%     
%  Outputs:
%        test_error: binary classification error of w on the data set (X, y)
%        this should be between 0 and 1. 
    [N, ~] = size(X);
    X = [ones(N,1) X]; % add an inital column of 1
    test_error = mean(sign(X*w) ~= y);
end

