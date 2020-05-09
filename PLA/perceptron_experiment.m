function [num_iters, bounds] = perceptron_experiment (N, d, num_samples)
%perceptron_experiment Code for running the perceptron experiment
%   Inputs: N is the number of training examples
%           d is the dimensionality of each example (before adding the 1)
%           num_samples is the number of times to repeat the experiment
%   Outputs: num_iters is the # of iterations PLA takes for each sample
%            bounds is the theoretical bound on the # of iterations
%              for each sample
%      (both the outputs should be num_samples long)
    num_iters = zeros(1, num_samples);
    bounds = zeros(1, num_samples);
    for i = 1:num_samples
        w_optimal = [0, rand(1, d)];
        w_optimal_norm = norm(w_optimal);
        D = ones(N, 1);
        D(:,2:d + 1) = 2 * rand(N, d) - 1;
        x_norm = sum(abs(D).^2,2).^(1/2);
        R = max(x_norm);
        Y = sign(D * w_optimal');
        Rho = min(abs(D * w_optimal'));
        D(:,d+2) = Y;
        [~, iter] = perceptron_learn(D);
        num_iters(i) = iter;
        bounds(i) = (R^2)*(w_optimal_norm^2)/(Rho^2);
    end
end