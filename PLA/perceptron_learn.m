function [w, iterations] = perceptron_learn(data_in)
%perceptron_learn Run PLA on the input data
%   Inputs: data_in: Assumed to be a matrix with each row representing an
%                    (x,y) pair, with the x vector augmented with an
%                    initial 1, and the label (y) in the last column
%   Outputs: w: A weight vector (should linearly separate the data if it is
%               linearly separable)
%            iterations: The number of iterations the algorithm ran for
    data_size = size(data_in);
    w_len = data_size(2) - 1;
    w = zeros(1, w_len); % initialize w as a zero vector
    data_x = data_in(:,1:w_len);
    data_y = data_in(:,data_size(2));
    iterations = 0;
    while 1 == 1
        iterations = iterations + 1;
        y_classified = sign(data_x * w');
        y_difference = y_classified - data_y;
        different_idx = find(y_difference);
        if isempty(different_idx) % all data correctly classified
            break;
        else % otherwise
            idx = different_idx(1); % pick a misclassified example
            w = w + data_y(idx)*data_x(idx,:);
        end
    end
end

