function [ A_est,F_est,G_est,Sigma_est,mu_est, progress_lik] = EM_estimate(A_init, F_init,G_init,Sigma_init,mu_init,...
    X_train,T,L,thresh,f,Dh,Dw,computeLogLik )
% Estimates DPLDA model parameters and if (computeLogLik) returns the
% value of the loglikelihood at every iteration to verify that it keeps
% increasing. Since it's expensive to do so, once sanity checks are
% performed, computeLogLik should be set to 0.

%It's optimized acording to the long'macro-video' implementation and
%thread-safe (parfor).

%OUTPUT: estimated parameters and progress of loglikelihood
%
%INPUT: 
%       A_init, F_init,G_init,Sigma_init,mu_init - initial guess for param
%       X - training data
%       I,J,T,f,Dh,Dw - hyperparameters
%       L,thresh - parameters responsible for duration of learning phase
%       coputeLogLik: if 1, we keep track of lerning progress by computing
%               loglik at every step and making sure it keeps increasing. 
%               Set to 0 for efficiency.


    takeTime=0; %setting this to 1 allows for tic-toc operations to identify bottlenecks
    if (takeTime)
    time=zeros(1,7);    %1: other
                        %2:E-step Offline filter
                        %3:E-step Offline smoothers (J-dependent) +
                        %  Offline2Online
                        %4:E-step Online
                        %4:B
                        %5:A
                        %6:Sigma
    end

    %% Intitalization

    progress_lik=zeros(1,L);
    disp '________EM Algorithm________'

    if (takeTime)
        tic
    end
    mu_est=mu_init;
    Sigma_est=(Sigma_init+Sigma_init')/2;
    A_est=A_init;
    G_est=G_init;
    F_est=F_init;
    % X_train=cell{numBuckets}. X_gal{j} contains videos of people for whom
    % excatly j training videos are available
    numBucketsTrain=length(X_train);
    nonEmptyBucketsTrain=[];
    for ind=1:numBucketsTrain
        if not(isempty(X_train{ind}))
            nonEmptyBucketsTrain=[nonEmptyBucketsTrain,ind];
        end
    end

    loglik_long=cell(1,length(X_train));

    %initialize so parfor won't get mad
    Sinv=NaN;
    log_detS=NaN;
    for b=nonEmptyBucketsTrain
        I=size(X_train{b},2);
        loglik_long{b}=zeros(I,1);
    end
    maxJ=max(nonEmptyBucketsTrain);
    model=ones(1,maxJ*T); %keeps track of time steps at which a transition 
                          %takes place within a macrovideo
    for j=1:maxJ-1
       model(j*T+1)=2; 
    end
    converged=0;
    l=1;
    old_loglik=-Inf;
    if (takeTime)
        time(1)=time(1)+toc;
        tic
    end
    %% Cycle ì, updating parameters at each iteration through ML-like criterion

    while l<=L && (converged==0) %either converged or reached maxNumber or iterations
        if (takeTime)
            tic
            parziale=0;
        end

        %Set up time-varying model: A, the state matrix, changes when we have a
        %transition from video 1 of person i to video 2 of person i
        [A_long,C_long,Sigma_long,Gamma_long,mu_long] = Utils.stackMatrices(F_est,G_est,A_est,Sigma_est,mu_est,1,T,Dh,Dw);

        A_long_=zeros(Dh+Dw,Dh+Dw,2);
        A_long_(:,:,1)=A_long;
        A_long_(:,:,2)=[eye(Dh),zeros(Dh,Dw); zeros(Dw,Dh),zeros(Dw,Dw)]; %transition matrix from a video to the following

        fprintf('\n')
        str= ['Iteration ',num2str(l),' out of (max) ',num2str(L)];
        disp (str);

        if (takeTime)
        time(1)=time(1)+toc;
        parziale=parziale+toc;
        tic;
        end

        fprintf('\n')
        disp('Estep');

        %offline Kalman filter to compute kalman gains and observation
        %covariance matrix
        if not(computeLogLik)
            [V,K] = Kalman.Offline_filter(A_long_,C_long,Gamma_long,Sigma_long, eye(Dw+Dh),maxJ*T,0,'model',model);
            disp('Computed V,K')
        else
            [V,K,Sinv,log_detS] = Kalman.Offline_filter(A_long_,C_long,Gamma_long,Sigma_long, eye(Dw+Dh),maxJ*T,1,'model',model);
            disp('Computed V,K Sinv and logDetS')
        end

        if (takeTime)
        time(2)=time(2)+toc;
        parziale=parziale+toc;
        tic
        end

        numB=cell(max(nonEmptyBucketsTrain),1);
        denB=cell(max(nonEmptyBucketsTrain),1);

        numA=cell(max(nonEmptyBucketsTrain),1);
        denA=cell(max(nonEmptyBucketsTrain),1);

        diagonal=cell(max(nonEmptyBucketsTrain),1);

        V_z_sum=cell(max(nonEmptyBucketsTrain),1); %this is to compute Sigma,
        %which has to be done after the parfor coz it needs C

        for b=nonEmptyBucketsTrain
            numB{b}=zeros(f,Dw+Dh);
            denB{b}=zeros(Dw+Dh);

            numA{b}=zeros(Dw);
            denA{b}=zeros(Dw);

            diagonal{b}=zeros(f,1);

            V_z_sum{b}=zeros(Dh+Dw,Dh+Dw);
        end

        for b=nonEmptyBucketsTrain
        % parfor b=nonEmptyBucketsTrain %it works!
            disp(num2str(b));
            I=size(X_train{b},2);
            X_long_detr=zeros(f,I,b*T);
            for j=1:b
                for t=1:T
                    X_long_detr(:,:,(j-1)*T+t)=X_train{b}(:,:,j,t);
                end
            end
            X_long_detr=X_long_detr-repmat(mu_est,1,I,b*T);
            [J_,V_z_smooth,VV_z_smooth]=Kalman.Offline_smoother(A_long_,Gamma_long,V(:,:,1:b*T),'model',model(1:b*T));

             E_z_filt=Kalman.Filter_Offline2Online_BATCH(X_long_detr,zeros(Dw+Dh,1),K,A_long_,C_long,'model',model);
             E_z_smooth{b}=Kalman.Smoother_Offline2Online_BATCH(A_long_,J_,E_z_filt,'model',model(1:b*T));
             if(computeLogLik)
                        loglik_long{b}=Kalman.Loglik_Offline2Online_BATCH(C_long,A_long_,Sinv,log_detS,zeros(Dh+Dw,1), E_z_filt,...
                                X_long_detr,'model',model);
                if (any(loglik_long{b}==-Inf)||any(loglik_long{b}==Inf) ||any(isnan(loglik_long{b})))
                    disp('Loglik is NaN or Inf')
                end

             end
            I=size(X_train{b},2);
            for j=1:b
              for t=1:T
                 numB{b}=numB{b}+X_long_detr(:,:,(j-1)*T+t)*E_z_smooth{b}(:,:,(j-1)*T+t)';
                 denB{b}=denB{b}+E_z_smooth{b}(:,:,(j-1)*T+t)*E_z_smooth{b}(:,:,(j-1)*T+t)'...
                            +V_z_smooth(:,:,(j-1)*T+t)*I;

                if (t>1)
                     numA{b}=numA{b}+E_z_smooth{b}(Dh+(1:Dw),:,(j-1)*T+t)*E_z_smooth{b}(Dh+(1:Dw),:,(j-1)*T+t-1)'...
                        +VV_z_smooth(Dh+(1:Dw),Dh+(1:Dw),(j-1)*T+t)*I;
                     denA{b}=denA{b}+E_z_smooth{b}(Dh+(1:Dw),:,(j-1)*T+t-1)*E_z_smooth{b}(Dh+(1:Dw),:,(j-1)*T+t-1)'...
                        +V_z_smooth(Dh+(1:Dw),Dh+(1:Dw),(j-1)*T+t-1)*I;
                end

                V_z_sum{b}=V_z_sum{b}+V_z_smooth(:,:,(j-1)*T+t)*I ;

              end
            end
        end
        num=zeros(f,Dw+Dh);
        den=zeros(Dw+Dh);
        for b=nonEmptyBucketsTrain
            num=num+numB{b};
            den=den+denB{b};
        end
        B_est=num/den;
        F_est=B_est(:,1:Dh);
        G_est=B_est(:,Dh+1:Dh+Dw);

        num=zeros(Dw);
        den=zeros(Dw);
        for b=nonEmptyBucketsTrain
            num=num+numA{b};
            den=den+denA{b};
        end
        A_est=num/den;

        %Once B is computed, I can compute Sigma, not before!
        diagonal=zeros(f,1);
        for b=nonEmptyBucketsTrain
        I=size(X_train{b},2);
        Ezz_sum=zeros(Dh+Dw,Dh+Dw);
            for j=1:b
              for t=1:T

                   diagonal=diagonal+dot((reshape(X_train{b}(:,:,j,t),f,I)-repmat(mu_est,1,I)),(reshape(X_train{b}(:,:,j,t),f,I)-repmat(mu_est,1,I)),2)...
                     -dot(2*B_est * E_z_smooth{b}(:,:,(j-1)*T+t),(reshape(X_train{b}(:,:,j,t),f,I)-repmat(mu_est,1,I)),2);
                 Ezz_sum=Ezz_sum+E_z_smooth{b}(:,:,(j-1)*T+t) * E_z_smooth{b}(:,:,(j-1)*T+t)';

              end
            end
            diagonal=diagonal+dot(B_est * (  Ezz_sum+V_z_sum{b} ), B_est,2);
        end

        N=Utils.totNumVids(X_train);
        Sigma_est=diag(diagonal)./(N*T);

         if (computeLogLik)
             progress_lik(l)=0;
             for b=nonEmptyBucketsTrain
                progress_lik(l)=sum(loglik_long{b});
             end
            converged=Utils.em_converged(progress_lik(l),old_loglik,thresh);
            if converged 
                progress_lik=progress_lik(1:l);
            end
            old_loglik=progress_lik(l);
         end
        l=l+1;
        if (takeTime)
        time(1)=time(1)+toc;
        fprintf('\n')
        disp( ['It took ',num2str(parziale),' s.']);
        tic
        end

    end
    % if (takeTime)
    %     keyboard;
    % end

end

