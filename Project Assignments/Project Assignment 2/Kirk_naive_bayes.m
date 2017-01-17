clear all;

% Loading training data sets
trainData = load ('train.data');
trainLabel = load ('train.label');

% Obtaining max dimensions for training data
% m = max(trainLabel);
% n = max(trainData(:,2));
% l = size(trainData,1);

% Initialization of empty matrices for faster calculations
DocWordCount = zeros(11269, 53975);

% Creating Doc*word_index Count matrix for train data set
for i = 1:1467345
    DocWordCount(trainData(i,1), trainData(i,2)) = trainData(i,3);
end;

% Clearing variables that won't be used again to speed up the program
clear trainData;

% Initialization of empty matrices for faster calculations 
WordIndexCountCat = zeros(20,53975);
totalWordsCat = zeros(20,1);
% prior = zeros(20,1);


for i = 1:20
    for j  = 1:11269
        % Adding to empty matrices if categories match with trainLabel
        % values
        if trainLabel(j) == i
        
        % Creating a sum of words for each category    
        WordIndexCountCat(i,:) = WordIndexCountCat(i,:) + DocWordCount(j, :);
        
        % Total words in a each Category 
        totalWordsCat(i,:) = totalWordsCat(i,:) + sum(DocWordCount(j,:));    
        end
    end 

end

% Clearing variables that won't be used again to speed up the program
clear DocWordCount trainLabel;

% Calculating Probability of Word given Categories
PWordCat = WordIndexCountCat./repmat(totalWordsCat,[1,53975]);

% Calculating log likelihood with Smoothing factor of 0.1
likelihood = log(((1 - 0.1)*PWordCat) + (0.1/53975));

% Calculating Prior
prior = log(totalWordsCat./sum(totalWordsCat));

% Clearing variables that won't be used again to speed up the program
clear WordIndexCountCat totalWordsCat;

% Loading Testing data
testData = load ('test.data');
testLabel = load('test.label');

% Creating Doc*word_index Count matrix for train data set
docWordCountTest = zeros(7505, 61188);
for i = 1:967874
    docWordCountTest(testData(i,1), testData(i,2)) = testData(i,3);
end;

%ignoring the new extra words
usedDocWordCountTest = docWordCountTest(:,1:53975);

% Calculating the Probability of Document given Categories
% Taking Log of likelihood^usedDocWordCountTest
PDocCat = likelihood * usedDocWordCountTest';

% Scaling Prior 
bigPrior = repmat(prior,[1,7505]);

% Calculating the Posterior
% Taking Log to calculate it as a Summation instead of Product 
posterior = bigPrior + PDocCat;

% Initialization of empty matrices for faster calculations
confidence = zeros(7505,1);
predictions = zeros(7505,1);

% Obtaining max values for posterior for every category
corrPred = 0;
for i = 1:7505
    [M,pred] = max(posterior(:,i));
    predictions(i,1) = pred;
    confidence(i,1) = M*100;
   if(pred == testLabel(i,1))
        corrPred = corrPred + 1;
    end; 
end



% Calculating accuracy of predictions
acc = (sum(testLabel == predictions)/length(testLabel))*100;

% Calculating misclassified predictions
wrong = testLabel(testLabel ~= predictions);

% Plotting misclassified predictions
hist(wrong,0.5)
xlabel('Categories')
ylabel('Wrong Predictions')