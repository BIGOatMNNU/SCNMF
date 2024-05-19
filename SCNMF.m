function [feature_slct, feature_weight, obj] = SCNMF(data,Y,lambda1,lambda2)
%2024_01_13

[n,d] = size(data);
[~,c] = size(Y);

LL = pdist2(Y',Y','cosine');
LL(isnan(LL)) = 0;

ker  = 'rbf'; %type of kernel function ('lin', 'poly', 'rbf', 'sam')
par  = 1*mean(pdist(data')); %parameter of kernel function
H = kernelmatrix(ker, par, data', data');

%optimization
iter=1;
eps = 1e-20;
W = ones(d,c)*.5;
while(1)
    
    HW = diag(1./max(sqrt(sum((W).*(W),2)),eps));
    W=W.*(data'*Y+2*lambda1*H*W)./max(data'*data*W+.5*HW*W+2*lambda1*(W*W'*W)+...
        lambda2*W*LL,eps);
    
    obj(iter)=(norm((data*W - Y), 'fro'))^2+sum(sqrt(sum((W).*(W),2)))+ ...
        lambda1*(norm((H-W*W'), 'fro'))^2+...
        lambda2*(trace(LL*W'*W));
    disp(iter+":"+obj(iter));
    if (iter>=2 && abs(obj(iter)-obj(iter-1))<=1e-2) || iter>1000
        break;
    end
    iter=1+iter;
end

feature_weight = sum(W, 2);
[~, feature_slct] = sort(feature_weight, 'descend');
end



