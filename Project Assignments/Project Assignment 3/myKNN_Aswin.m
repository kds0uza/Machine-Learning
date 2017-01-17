function [prediction, bestk, errors] = myKNN(X, Y, Xtest, k)
    errors = zeros(1, size(k, 2));
    for ktest = 1:(size(k, 2)) %select number of neighbors
        errors(ktest) = 0;
        for i = 1:size(X, 1) %leave one out
            testX = repmat(X(i, :), size(X, 1), 1);
            testY = Y(i);
            distV = [sqrt(sum((testX-X).^2, 2)), Y];
            [~,sortV] = sort(distV(:,1)) ;
            distV = distV(sortV, :);
            pred = mode(distV(2:(k(ktest)+1),2));
            if pred ~= testY
                errors(ktest) = errors(ktest) + 1; %increment if prediction is wrong
            end
        end
    end
        [~, minerror] = min(errors);
        bestk = k(minerror);
        prediction = zeros(size(Xtest, 1), 1);
      for i = 1:size(Xtest, 1)
          testX = repmat(Xtest(i, :), size(X, 1), 1);
          distV = [sqrt(sum((testX-X).^2, 2)), Y];
          [~,sortV] = sort(distV(:,1)) ;
          distV = distV(sortV, :);
          prediction(i) = mode(distV(2:(bestk+1),2));
      end 
    end
    
    