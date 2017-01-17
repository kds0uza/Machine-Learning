clear all;
trainData = load ('train.data');
trainLabel = load ('train.label');

DocumentTermMatrix = zeros(11269, 53975);

for i = 1:1467345
    DocumentTermMatrix(trainData(i,1), trainData(i,2)) = trainData(i,3);
end;

clear trainData
CountWordForClass = zeros(20,53975);
SumOfTotalWordsForClass = zeros(20,1);
SumEachClass = zeros(20,1);
classCount = 1;
wordCount = 0;
for j  = 1:11269
    
    CountWordForClass(trainLabel(j),:) = CountWordForClass(trainLabel(j),:) + DocumentTermMatrix(j, :);
    
    SumOfTotalWordsForClass(trainLabel(j)) = SumOfTotalWordsForClass(trainLabel(j)) + sum(DocumentTermMatrix(j,:));
    
    SumEachClass(trainLabel(j)) = SumEachClass(trainLabel(j)) + 1;

end;
clear DocTermMatrix trainLabel

% ProbablityPriorClass = zeros(20,1);

%ProbablityPriorClass = SumEachClass/11269;

ProbabilityWordGivenClass = zeros(53975, 20);

for i = 1:53975
    for j = 1:20
        ProbabilityWordGivenClass(i,j) = (CountWordForClass(j,i) + 1) / (SumOfTotalWordsForClass(j) + 53975);
    end;
end;

clear CountWordForClass
testData = load ('test.data');


DocumentTermMatrixTest = zeros(7505, 61188);
for i = 1:967874
    DocumentTermMatrixTest(testData(i,1), testData(i,2)) = testData(i,3);
end;

clear testData
testLabel = load ('test.label');
CountWordForClassTest = zeros(20,61188);
for j  = 1:7505
    
    CountWordForClassTest(trainLabel(j),:) = CountWordForClassTest(trainLabel(j),:) + DocumentTermMatrixTest(j, :);
    
end;
clear DocumentTermMatrixTest trainLabel
ProbablityDocumentGivenClass = zeros(61188, 20);

for i = 1:61188
    for j = 1:20
        ProbablityDocumentGivenClass(i,j) = ProbabilityWordGivenClass(i,j)^CountWordForClassTest(j,i);
    end;
end;

ProbablityClass = zeros(20,1);


    
