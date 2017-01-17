function fun = objFun(alpha,trainLabels,kernel,C)
delta = (alpha/C);
bigOne = ones(size(delta,1),1);
bigG1 = bsxfun(@times,delta,log(delta));
% bigG1 = delta'*log(delta);
bigG2 = bsxfun(@times,(bigOne - delta),log(bigOne - delta));
% bigG2 = (bigOne - delta)'*log(bigOne - delta);
bigG3 = bigG1 +bigG2;
bigG = sum(bigG3);
bigAlpha = (1/2)*(alpha'*(trainLabels'*trainLabels.*kernel)*alpha);
fun = bigAlpha + bigG;
end