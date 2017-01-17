 clear all;

% Loads Training Sets
trainData = load ('train.data');
trainLabel = load ('train.label');
m = size(trainLabel,1);
n = max(trainData(:,2));
l = size(trainData,1);
clear trainLabel;

%Allocates and creates the Document-Term matrix for the training set
docWordCount = zeros(m, n);
for i = 1:l
    docWordCount(trainData(i,1), trainData(i,2)) = trainData(i,3);
end;

clear trainData;

trainLabel = load ('train.label');
wordsInCat = zeros(20,53975);
totalWordsInCat = zeros(20,1);
docsInCat = zeros(20,1);
prior = zeros(20,1);

for i = 1:20
for j  = 1:size(trainLabel,1)
    if trainLabel(j) == i
    %Sum of each word belonging to a particular class 
    wordsInCat(i,:) = wordsInCat(i,:) + docWordCount(j, :);
    
    %Sum of total words in a particular class
    totalWordsInCat(i,:) = totalWordsInCat(i,:) + sum(docWordCount(j,:));
    
%     %Calculates the number of documents belonging to each class
    docsInCat(i) = docsInCat(i) + 1;
    end
    prior(i) = log(docsInCat(i)/m);
end
end

 %Calculates the number of documents belonging to each class
% for i = 1:20
%     indexes = find(trainLabel(j) == i);
%     docsInCat(i) = docWordCount(max(indexes),1);
%     prior(i) = log(docsInCat(i)/m);
% end

% prior = log(SumEachClass/11269);
clear docsInCat trainLabel docWordCount;

probOfWordGivenCat = zeros(20,n);
%Parameters used for the smoothing to avoid zero probablities
% smoothFactor = 1-0.1;
% smoothAdd = 0.1/n;
for i = 1:53975
    for j = 1:20
        %Calculate the probablity of each word given each class
        probOfWordGivenCat(i,j) = log(((1 - 0.1)*(wordsInCat(j,i) / totalWordsInCat(j))) + (0.1/n));
    end
end

clear wordsInCat totalWordsInCat;

%Loads Test data
testData = load ('test.data');
testLabel = load('test.label');
m = size(testLabel,1);
n = max(testData(:,2));
l = size(testData,1)
%Calculates the Document-Term matrix for the test data
docWordCountTest = zeros(7505, 61188);
for i = 1:l
    docWordCountTest(testData(i,1), testData(i,2)) = testData(i,3);
end

probOfDocGivenCat = zeros(1, 61188);
ProbablityClass = zeros(20,7505);

for i = 1:20
    for j = 1:7505
        %Calculates probablities of test document given category in each loop
        for k = 1:53975
             
            probOfDocGivenCat(1,k) = (docWordCountTest(j,k) * probOfWordGivenCat(k,i));
            
             
            
        end
         
        
    end
end

% Calculates probablity of the class given a document in each loop
for i = 1:20
    for j = 1:m
    ProbablityClass(i,j) = prior(i) + sum(probOfDocGivenCat);
    end
end
clear ProbablityPriorClass ProbablityDocumentGivenClass ProbabilityWordGivenClass;

%Calculates which class has the max probablity for each document to make a
%prediction and calculate the number of correct predictions by comparing
%our predictions with the ones in the test label
% predictions = zeros(7505,1);

correct = 0;
for i = 1:7505
    [M,B] = max(ProbablityClass(:,i));
    if(B == testLabel(i,1))
        correct = correct + 1;
    end
end


corr = sum(testLabel == B); 



%Once the number of correct predictions are calculated, we calculate the
%accuracy of our classfier.
accuracy = (correct/7505)*100;
    
