function [ oobErr ] = BaggedTrees( X, Y, numBags )
%BAGGEDTREES Returns out-of-bag classification error of an ensemble of
%numBags CART decision trees on the input dataset, and also plots the error
%as a function of the number of bags from 1 to numBags
%   Inputs:
%       X : Matrix of training data
%       Y : Vector of classes of the training examples
%       numBags : Number of trees to learn in the ensemble
%
%   You may use "fitctree" but do not use "TreeBagger" or any other inbuilt
%   bagging function
    [N,P] = size(X);
    idx = randi(N,N,numBags); % Store N by numBags random indices
    ts = cell(numBags,1); % Store #numBags trees
    for j = 1:numBags
        % Generate random data set for each tree
        Xm = zeros(N,P);
        Ym = zeros(N,1); 
        for i = 1:N
            Xm(i,:) = X(idx(i,j),:);
            Ym(i) = Y(idx(i,j));
        end
        % Train a tree and store it
        ts{j} = fitctree(Xm,Ym);
    end

    ps = zeros(N,numBags); % Store predictions of every tree on every data
    for i = 1:N
        for j = 1:numBags
            ps(i,j) = predict(ts{j},X(i,:));
        end
    end

    oobErrs = zeros(numBags,1); % Store an array of oob
    for j = 1:numBags
        agg = zeros(N,1); % We need to aggregate at most N data points
        valid_agg = 0;
        for i = 1:N % For every Gi-
            % Gi- can aggregate at most j trees (index stored in gs)
            gs = zeros(j,1); 
            valid_g = 0;
            for k = 1:j % We only look at the first j trees
                % If the tree k doesn't use i
                if isempty(find(idx(:,k)==i, 1)) 
                    valid_g = valid_g + 1;
                    gs(valid_g) = k; % Store the index of the tree
                end
            end
            
            % If Gi- has no tree to aggregate (all trees use it)
            if valid_g == 0 
                continue;
            end
            
            % Store the result predicted by every tree in gs
            pred = 0; 
            for k = 1:valid_g
                pred = pred + ps(i,gs(k));
            end
            res = pred/valid_g; % mean of pred
            valid_agg = valid_agg + 1;
            % Wrong if the mean and the true label has difference greater
            % than 1
            agg(valid_agg) =  abs(res - Y(i)) > 1; 
        end
        oobErrs(j) = sum(agg)/valid_agg;
    end
    
    % Plot OOB Error as a function of the number of bags
    figure();
    plot(oobErrs);
    xlabel('Number of bags')
    ylabel('Out-of-bag error')
    title('OOB Error as a function of the number of bags')

    oobErr = oobErrs(numBags);
end
