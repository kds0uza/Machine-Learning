% clear all;

% Loads Training Sets
trainData = load ('train.data');
trainLabel = load ('train.label');

%Allocates and creates the Document-Term matrix for the training set
DocumentTermMatrix = zeros(11269, 53975);
for i = 1:1467345
    DocumentTermMatrix(trainData(i,1), trainData(i,2)) = trainData(i,3);
end;

clear trainData;
 
CountWordForClass = zeros(20,53975);
SumOfTotalWordsForClass = zeros(20,1);
SumEachClass = zeros(20,1);
for j  = 1:11269
    
    %Sum of each word belonging to a particular class 
    CountWordForClass(trainLabel(j),:) = CountWordForClass(trainLabel(j),:) + DocumentTermMatrix(j, :);
    
    %Sum of total words in a particular class
    SumOfTotalWordsForClass(trainLabel(j)) = SumOfTotalWordsForClass(trainLabel(j)) + sum(DocumentTermMatrix(j,:));
    
    %Calculates the number of documents belonging to each class
    SumEachClass(trainLabel(j)) = SumEachClass(trainLabel(j)) + 1;

end;

ProbablityPriorClass = log(SumEachClass/11269);
clear SumEachClass trainLabel DocumentTermMatrix trainLabel;

ProbabilityWordGivenClass = zeros(53975, 20);
%Parameters used for the smoothing to avoid zero probablities
smoothMul = 1-0.1;
smoothAdd = 0.1/53975;
for i = 1:53975
    for j = 1:20
        %Calculate the probablity of each word given each class
        ProbabilityWordGivenClass(i,j) = log((smoothMul*(CountWordForClass(j,i) / SumOfTotalWordsForClass(j))) + smoothAdd);
    end;
end;

clear CountWordForClass SumOfTotalWordsForClass;

%Loads Test data
testData = load ('test.data');

%Calculates the Document-Term matrix for the test data
DocumentTermMatrixTest = zeros(7505, 61188);
for i = 1:967874
    DocumentTermMatrixTest(testData(i,1), testData(i,2)) = testData(i,3);
end;

ProbablityDocumentGivenClass = zeros(1, 61188);
ProbablityClass = zeros(20,7505);

for i = 1:20
    for j = 1:7505
        
        %Calculates probablities of test document given class in each loop
        for k = 1:53975
            ProbablityDocumentGivenClass(1,k) = (DocumentTermMatrixTest(j,k) * ProbabilityWordGivenClass(k,i));
        end;
        % Calculates probablity of the class given a document in each loop 
        ProbablityClass(i,j) = ProbablityPriorClass(i) + sum(ProbablityDocumentGivenClass);
    end;
end;

clear ProbablityPriorClass ProbablityDocumentGivenClass ProbabilityWordGivenClass;

%Calculates which class has the max probablity for each document to make a
%prediction and calculate the number of correct predictions by comparing
%our predictions with the ones in the test label
predictions = zeros(7505,1);
testLabel = load('test.label');
correctPredictions = 0;
for i = 1:7505
    [M,B] = max(ProbablityClass(:,i));
    if(B == testLabel(i,1))
        correctPredictions = correctPredictions + 1;
    end;
end;

%Once the number of correct predictions are calculated, we calculate the
%accuracy of our classfier.
accuracy = (correctPredictions/7505)*100;
    
