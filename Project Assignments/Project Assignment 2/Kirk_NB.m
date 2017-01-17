clear all;
trainData = load ('train.data');


DocWordCount = zeros(11269, 53975);

%Creating Doc*Word_Index Count matrix for training data set
for i = 1:1467345
    DocWordCount(trainData(i,1), trainData(i,2)) = trainData(i,3);
end;

clear trainData;

%Initializing all matrices with 0s
WordIndexCountForClasses = zeros(20,53975);
TotalWordsForClasses = zeros(20,1);
SumEachClass = zeros(20,1);
PriorClass = zeros(20,1);

%Loading training data labels
trainLabel = load ('train.label');

%Creating Class * Word_Index Count matrix and Total Words in each Class Column
%vector
for i = 1:20
for j  = 1:11269
    if trainLabel(j) == i
        
        WordIndexCountForClasses(i,:) = WordIndexCountForClasses(i,:) + DocWordCount(j, :);
    
        TotalWordsForClasses(i,:) = TotalWordsForClasses(i,:) + sum(DocWordCount(j,:));
    
        SumEachClass(i) = SumEachClass(i) + 1;
        
    end
end
PriorClass(i) = (SumEachClass(i)/11269);
end

% PriorClass = (SumEachClass/11269);
clear SumEachClass trainLabel DocWordCount;

ProbabilityWordGivenClass = zeros(53975, 20);
smoothMul = 1-0.01;
smoothAdd = 0.01/53975;
for i = 1:53975
    for j = 1:20
%         ProbabilityWordGivenClass(i,j) = (CountWordForClass(j,i) / SumOfTotalWordsForClass(j)) + smoothAdd;
        ProbabilityWordGivenClass(i,j) = ((WordIndexCountForClasses(j,i) + 1) / (TotalWordsForClasses(j) + 53975));
    end;
end;

clear WordIndexCountForClasses TotalWordsForClasses;

testData = load ('test.data');

DocumentTermMatrixTest = zeros(7505, 61188);
for i = 1:967874
    DocumentTermMatrixTest(testData(i,1), testData(i,2)) = testData(i,3);
end;

ProbablityDocumentGivenClass = ones(1, 61188);
ProbablityClass = zeros(20,7505);

for i = 1:20
    for j = 1:7505
        
         for k = 1:53975
             ProbablityDocumentGivenClass(1,k) = ProbabilityWordGivenClass(k,i)^DocumentTermMatrixTest(j,k);
         end;
         
        ProbablityClass(i,j) = (PriorClass(i) * prod(ProbablityDocumentGivenClass(1,:)));
    end;
end;

clear ProbablityPriorClass ProbablityDocumentGivenClass ProbabilityWordGivenClass;

predictions = zeros(7505,1);

for i = 1:7505
    [M,B] = max(ProbablityClass(:,i));
    predictions(i,1) = B;
end;

testLabel = load('test.label');
correctPredictions = 0;
for i = 1:7505
    if predictions(i,1) == testLabel(i,1)
        correctPredictions = correctPredictions + 1;
    end;
end;

accuracy = (correctPredictions/7505)*100;
    
