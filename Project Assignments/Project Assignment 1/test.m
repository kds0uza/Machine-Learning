data = load('data_slump_test.dat');
noutput = 3;

k = 3;
indx = crossvalind('Kfold', length(data), k);
l = numel(indx(indx<k));

trainX = zeros(l,size(data,2));
testX = zeros(length(data)-l,size(data,2));

trainX_indx = zeros(length(data),1);
testX_indx = zeros(length(data),1);

% myregression(trainX, testX, noutput)
for i = 1:length(data)
    if indx(i) < k
       trainX_indx(i) = i;
       testX_indx(i) = 0;
    else
        trainX_indx(i) = 0;
        testX_indx(i) = i;
    end
end

trainX_indx = trainX_indx(trainX_indx~=0);
testX_indx = testX_indx(testX_indx~=0);

for i = 1:length(trainX)
    trainX(i,:) = data(trainX_indx(i),:);
end

for i = 1:length(testX)
    testX(i,:) = data(testX_indx(i),:);
end

%trainX=trainX(:,2:size(trainX,2));
out = testX(:,size(testX,2));
testX = testX(:,1:size(data,2)-noutput);
%testX = testX(:,2:size(data,2)-noutput);