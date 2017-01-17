function [ pred ] = myregression( trainX,testX,noutput)

% X = [ones(length(trainX),1) trainX(:,1:size(trainX,2)-noutput);];
% t = trainX(:,size(trainX,2)-noutput+1:size(trainX,2));
% w = inv(X'*X)*X'*t
% pred = [ones(length(testX),1) testX]*w;

warning('off','all');

% Cross-Validation

k = 5;
testFoldSize = floor(length(trainX)/k);
foldParts = 1:testFoldSize:length(trainX);
min = intmax('int32');

for (fold = 1:k)
    testRows = foldParts(fold) : (foldParts(fold)+ testFoldSize-1);
    testCV = testX(testRows, :);
    trainRows = setdiff((1:length(trainX)),testRows);
    trainCV = trainX(trainRows, :);
    
    X = trainCV(:,1:size(trainCV,2)-noutput);
    %     Feature scaling
    X = (X-mean(X))./sqrt(var(X));
% X =[ X  ];

    X = [ones(length(X),1) X];
    t = trainCV(:,size(trainCV,2)-noutput+1:size(trainCV,2));

% w = inv(X'*X)*X'*t;
%inv(A)*b can be slower and accurate so using X\t
    w = X\t;

%     Feature scaling
    testCV = (testCV - mean(testCV))./sqrt(var(testCV));
    % testX = [testX   ];
    predCV = [ones(length(testCV),1) testCV]*w;
    predCV = real(predCV);
    
    Y = trainX(testRows,6);
    error = norm(predCV - Y);
    
    if min > error
        wCV = w;
%         lowestErrorFold = fold;
        min = error;
    end
end   

testX = (testX - mean(testX))./sqrt(var(testX));
pred = [ones(length(testX),1) testX]*wCV;
pred = real(pred);

end













