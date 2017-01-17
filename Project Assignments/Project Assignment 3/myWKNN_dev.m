function [ prediction, bestlambda ] = myWKNN_dev( X, Y, Xtest, lambda)

numTrain = size(X, 1);
numLambda = size(lambda,1);
predictedClassCV = zeros(1,numTrain);
pred = zeros(numTrain,1);
errors = zeros(1, numLambda);

for m = 1:numLambda
    for i = 1:numTrain
       % numTrainCV = numTrain-1;
        testCV = X(i, :);
        
       % yTestCV = Y(i,1);
        
        %All elements except testCV element 
        trainCV = X;
        YtrainCV = Y;
        trainCV(i, :) = [];
        YtrainCV(i) = [];
        
        distances = distance2(trainCV, testCV, 2);
        
        [~, nearestIndices] = sort(distances);
        nearestClassValues = YtrainCV(nearestIndices(1:959), 1);
        w = exp(-lambda(m)*distances(nearestIndices(1:959),1));
        
        for j = 1:3
            delta = (nearestClassValues==j);
            predictedClassCV(j) = sum(w .* delta)/sum(w);
        end
        
        [~, pred(i,1)] = max(predictedClassCV);
    end
    
   errors(m) = sum(pred ~= Y);

end        
    [~, leastError] = min(errors);
     
    bestlambda = lambda(leastError);
    bestk = 29;
    
    %Testing
    numTest = size(Xtest,1);
    predictedClassTest = zeros(1, numTest);
    testDistances = distance2(Xtest, X, 2);
    
    [~, sortedTestIndices] = sort(testDistances,2);
    
    prediction = zeros(numTest, 1);
    
    for i = 1:numTest
        nearestTestClassValues = Y(sortedTestIndices(i, 1:bestk), 1);
        wTest = exp(-bestlambda*testDistances(i, sortedTestIndices(i, 1:bestk)));
        
        for j = 1:3
            deltaTest = (nearestTestClassValues==j);
            predictedClassTest(j) = sum(wTest' .* deltaTest)/sum(wTest);
        end
        
        [~, prediction(i,1)] = max(predictedClassTest);
    end
end

function [D]=distance2(X,Y,A)
% computes sq. A-norm distance between two D-dimensional vectors (rows of X
% and Y)
% X - [Nx x D] matrix
% Y - [Ny x D] matrix
% A - [D x D] matrix (optional input)
if(nargin<3)
    A=eye(size(X,2));
end;

    D=bsxfun(@plus,bsxfun(@plus,-2*X*A*Y',sum(X*A.*X,2)),[sum(Y*A.*Y,2)]');
end