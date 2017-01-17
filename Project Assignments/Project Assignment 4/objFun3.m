function fun3 = objFun3(alpha,trainLabels,kernel,C)
delta = (alpha/C);
bigG = delta.*log(delta) + (1 - delta).*log(1 - delta);
bigAlpha = 0.5*(delta*delta').*(trainLabels*trainLabels').*kernel + C*sum(bigG);
fun3 = sum(bigAlpha(:));
end