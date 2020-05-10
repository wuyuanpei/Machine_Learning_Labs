clear all;

% load data
fprintf('Working on the one-vs-three problem...\n\n');
load zip.train;
subsampleTrain = zip(find(zip(:,1)==1 | zip(:,1) == 3),:);
Yt = subsampleTrain(:,1);
Xt = subsampleTrain(:,2:257);

load zip.test
subsample = zip(find(zip(:,1)==1 | zip(:,1) == 3),:);
Y = subsample(:,1);
X = subsample(:,2:257);

% A single decision tree
ct = fitctree(Xt,Yt);
Prediction = predict(ct,X);
Compare = not(Prediction == Y);
test_error = mean(Compare);

fprintf('The test error of a single decision tree is %.4f\n', test_error);

% Ensemble of 200 trees

numBags = 200;

[N,P] = size(Xt);
idx = randi(N,N,numBags); % Store N by numBags random indices

[M,~]=size(Y);
Prediction = zeros(M,numBags);

for j = 1:numBags
    Xm = zeros(N,P);
    Ym = zeros(N,1);    
    for i = 1:N
        Xm(i,:) = Xt(idx(i,j),:);
        Ym(i) = Yt(idx(i,j));
    end
    t = fitctree(Xm,Ym);
    Prediction(:,j) = predict(t,X);
end

majority = mean(Prediction,2);
test_error = mean(abs(majority - Y) > 1);

fprintf('The test error of an ensemble of 200 decision trees is %.4f\n\n', test_error);


clear all;

% load data
fprintf('Working on the three-vs-five problem...\n\n');
load zip.train;
subsampleTrain = zip(find(zip(:,1)==3 | zip(:,1) == 5),:);
Yt = subsampleTrain(:,1);
Xt = subsampleTrain(:,2:257);

load zip.test
subsample = zip(find(zip(:,1)==3 | zip(:,1) == 5),:);
Y = subsample(:,1);
X = subsample(:,2:257);

% A single decision tree
ct = fitctree(Xt,Yt);
Prediction = predict(ct,X);
Compare = not(Prediction == Y);
test_error = mean(Compare);

fprintf('The test error of a single decision tree is %.4f\n', test_error);

% Ensemble of 200 trees

numBags = 200;

[N,P] = size(Xt);
idx = randi(N,N,numBags); % Store N by numBags random indices

[M,~]=size(Y);
Prediction = zeros(M,numBags);

for j = 1:numBags
    Xm = zeros(N,P);
    Ym = zeros(N,1);    
    for i = 1:N
        Xm(i,:) = Xt(idx(i,j),:);
        Ym(i) = Yt(idx(i,j));
    end
    t = fitctree(Xm,Ym);
    Prediction(:,j) = predict(t,X);
end

majority = mean(Prediction,2);
test_error = mean(abs(majority - Y) > 1);

fprintf('The test error of an ensemble of 200 decision trees is %.4f\n', test_error);






