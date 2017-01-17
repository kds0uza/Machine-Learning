clc 
close all
clear

train_data = load('train.data');
train_label = load('train.label');

occ_matrix = zeros(max(train_data(:,1)),max(train_data(:,2)));

for i = 1:size(train_data,1)
    occ_matrix(train_data(i,1),train_data(i,2))=train_data(i,3);
end

m = max(train_data(:,2));
doc_num = max(train_data(:,1));
l = max(train_label);
n_k = zeros(l,1);
prior = zeros(l,1);
numerator = zeros(l,m);
den = zeros(l,1);
pwgc = zeros(m,l);
% finding prior

for i = 1:l
    n_k(i) = numel(find(train_label==i));
    prior(i) = n_k(i)/max(train_data(:,1));
end

for i = 1:l
    ndx = find(train_label == i);
    numerator(i,:) = sum(occ_matrix(ndx(1):ndx(end),:));
    den = sum(numerator,2);
end
clear ooc_matrix
for i = 1:m
    for j = 1:l
        pwgc(i,j) = 0.9*numerator(j,i)/den(j) + 0.1/m;
    end
end
clear numerator den;
% testing

test_data =  load('test.data');
test_label = load('test.label');

occ_matrixTest = zeros(max(test_data(:,1)),max(test_data(:,2)));
for i = 1:size(test_data,1)
    occ_matrix(test_data(i,1),test_data(i,2))=test_data(i,3);
end

