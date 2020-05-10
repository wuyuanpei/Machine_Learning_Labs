clear all

% 1 vs 3
fprintf('Working on the one-vs-three problem...\n\n');
load zip.train;
subsampleTrain = zip(find(zip(:,1)==1 | zip(:,1) == 3),:);
y_tr = subsampleTrain(:,1) - 2;
X_tr = subsampleTrain(:,2:257);

load zip.test
subsample = zip(find(zip(:,1)==1 | zip(:,1) == 3),:);
y_te = subsample(:,1) - 2;
X_te = subsample(:,2:257);

n_trees = 200;
train_errs = zeros(n_trees,1);
test_errs = zeros(n_trees,1);
for i = 1:n_trees
    [train_errs(i), test_errs(i)] = AdaBoost(X_tr, y_tr, X_te, y_te, i);
end

% Plot Error as a function of the number of weak hypotheses
figure();
hold on
plot(train_errs);
plot(test_errs);
xlabel('Number of weak hypotheses');
ylabel('Error');
title('Error as a function of the number of weak hypotheses');
legend('Training set error','Test set error');



% 3 vs 5
clear all

fprintf('Working on the three-vs-five problem...\n\n');
load zip.train;
subsampleTrain = zip(find(zip(:,1)==3 | zip(:,1) == 5),:);
y_tr = subsampleTrain(:,1) - 4;
X_tr = subsampleTrain(:,2:257);

load zip.test
subsample = zip(find(zip(:,1)==3 | zip(:,1) == 5),:);
y_te = subsample(:,1) - 4;
X_te = subsample(:,2:257);

n_trees = 200;
train_errs = zeros(n_trees,1);
test_errs = zeros(n_trees,1);
for i = 1:n_trees
    [train_errs(i), test_errs(i)] = AdaBoost(X_tr, y_tr, X_te, y_te, i);
end

% Plot Error as a function of the number of weak hypotheses
figure();
hold on
plot(train_errs);
plot(test_errs);
xlabel('Number of weak hypotheses');
ylabel('Error');
title('Error as a function of the number of weak hypotheses');
legend('Training set error','Test set error');

