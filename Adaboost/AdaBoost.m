function [ train_err, test_err ] = AdaBoost( X_tr, y_tr, X_te, y_te, n_trees )
%AdaBoost: Implement AdaBoost using decision stumps learned
%   using information gain as the weak learners.
%   X_tr: Training set
%   y_tr: Training set labels
%   X_te: Testing set
%   y_te: Testing set labels
%   n_trees: The number of trees to use
    [N,~] = size(X_tr);
    [M,~] = size(X_te);
    weights = ones(N,1)/N; % Store weights

    alphas = zeros(n_trees,1); % Store all alphas

    row_predictions_tr = zeros(N,n_trees);
    row_predictions_te = zeros(M,n_trees);

    for t = 1:n_trees
        tree = fitctree(X_tr,y_tr,... 
                    'MaxNumSplits',1,...
                    'SplitCriterion', 'deviance',...
                    'Weights',weights); % Learn a decision stump
        predictions = predict(tree,X_tr);
        error = sum((not(predictions==y_tr)).*weights);
        alphas(t) = 0.5*log((1-error)/error);
        weights = weights .* exp(- y_tr * alphas(t) .* predictions);
        weights = weights / sum(weights); % Update weights
        row_predictions_tr(:,t) = predict(tree, X_tr);
        row_predictions_te(:,t) = predict(tree, X_te);
    end

    results_tr = sign(row_predictions_tr * alphas);
    results_te = sign(row_predictions_te * alphas);

    train_err = sum(not(results_tr == y_tr))/N;
    test_err = sum(not(results_te == y_te))/M;

end

