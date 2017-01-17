function [prediction, bestlambda] = myWKNN(X, Y, Xtest)
lambda = [0.01, 0.05, 0.0746, 0.1, 0.5, 0.75, 1];
errors = zeros(1, size(lambda, 2));
for lambdatest = 1:length(lambda) %select number of neighbors
        errors(lambdatest) = 0;
        for i = 1:size(X, 1) %leave one out
            testX = repmat(X(i, :), size(X, 1), 1);
            testY = Y(i);
            distV = [sum((testX-X).^2, 2), Y];
            w = exp(-lambda(lambdatest)*distV(:, 1));
            distV(:, 1) = w;
            labels = unique(Y);
            predprob = zeros(length(labels), 2);
            for l = 1:length(labels)
                predprob(l, :) = [labels(l), sum(distV((distV(:,2)==labels(l))), 1)/sum(w)];
            end
            [~, predind] = max(predprob(:, 2));
            pred = predprob(predind, 1);
            if pred ~= testY
                errors(lambdatest) = errors(lambdatest) + 1; %increment if prediction is wrong
            end
        end
end
        [~, minerror] = min(errors);
        bestlambda = lambda(minerror);
        %build predictions
        prediction = zeros(size(Xtest, 1), 1);
        for i = 1:size(Xtest, 1) %leave one out
            testX = repmat(Xtest(i, :), size(X, 1), 1);
            distV = [sum((testX-X).^2, 2), Y];
            w = exp(-bestlambda*distV(:, 1));
            distV(:, 1) = w;
            predprob = zeros(length(labels), 2);
            for l = 1:length(labels)
                predprob(l, :) = [labels(l), sum(distV((distV(:,2)==labels(l))), 1)/sum(w)];
            end
            [~, predind] = max(predprob(:, 2));
            prediction(i) = predprob(predind, 1);
        end
