function U = myGPCA(Z, K, flag1)

% c = 100;
% ni = 3;
% 
% n = c*ni;
% p = n-c;
% M = create_M_matrix(ni*ones(1, c));
% L = eye(length(M))-M;

%Z: N_feat x N_sampl

if size(Z, 1)<size(Z, 2) 
    B = Z*(Z');
    %B = (Z*M)*(Z*M)';
    [evec, eval] = F_Eigen_f(B);
    %[evec, eval] = eigs(B, size(Z, 1), 'LM');
    p = min([length(eval), K, size(Z, 1), size(Z, 2)]);
    %if flag1 == 1
    if flag1 == 0
        U = evec(:,1:p);
    else
        U = evec(:,1:p)*diag(eval(1:p).^(-1/2));
    end
else
    %B = M*(Z'*Z)*M;
    B = (Z'*Z);
    [evec, eval] = F_Eigen_f(B);
    %[evec, eval] = eigs(B, size(Z, 2), 'LM');
    a = 2;
    if flag1 == 1
        a = 1;
    end   
    p = min([length(eval), K, size(Z, 1), size(Z, 2)]);
    U = Z*evec(:,1:p)*diag(eval(1:p).^(-1/a));
end