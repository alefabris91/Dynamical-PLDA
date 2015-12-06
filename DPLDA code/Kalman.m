classdef Kalman
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % functions related to Kalman inference
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    methods(Static)
        
        function [V,K,varargout] = Offline_filter(A, C, Q, R, init_V,T,computeLogDet, varargin)
        % INPUTS:
        % A - the system matrix
        % C - the observation matrix 
        % Q - the system covariance 
        % R - the observation covariance
        % init_V - the initial state covariance 
        % T - length of filtering horizon 
        % computeLogDet - binary val that determines whether we need to compute the
        %        inverse of the innovation covariance matrix and the logarithm of
        %        its determinant.
        %
        % OPTIONAL INPUTS (string/value pairs [default in brackets])
        % 'model' - allows for a time-varying model wherein A=A(t).  
        %     model(t)=m means use params from model m at time t [ones(1,T]
        %     In this case, matrix A takes an additional final dimension,
        %     i.e., A(:,:,m).
        % 'use_prev' - binary val. If equal to 1 exploits init_V in a different
        %       way, basically considering it as an actual estimate of the previous
        %       variance, not as a prior on initial process variance.
        %
        % OUTPUTS (where X is the hidden state being estimated)
        % V(:,:,t) = Cov[X(:,t) | y(:,1:t)]
        % K(:,:,t) = Kalman gain at time t
        % Sinv(:,:,t) = inv(Covar( y(:,t)|y(:,1:t-1) )) 
        % log_detS(:,t) = log[ det(Covar( y(:,t)|y(:,1:t-1) )) ]

            measureTime=0;
            if (measureTime)
                time=zeros(1,2);    %1:update
                                    %2:anything else
                tic
            end
            os  = size(C,1); %size of observations
            ss = size(A,1); % size of state space

            % set default params
            model = ones(1,T);
            usePrevResults=0;

            args = varargin; %parse optional arguments
            nargs = length(args);
            for i=1:2:nargs
              switch args{i}
               case 'model', model = args{i+1};
                  case 'use_prev', usePrevResults=args{i+1};
               otherwise, error(['unrecognized argument ' args{i}])
              end
            end

            if (computeLogDet)
                Sinv=zeros(os,os,T);
                log_detS=zeros(1,T);
            end
            K=zeros(ss,os,T);
            V = zeros(ss, ss, T);
            if (measureTime)
            time(2)=time(2)+toc;
            end
            for t=1:T
                if (measureTime)
                tic
                end
              m = model(t);
              if t==1
                prevV = init_V;
                initial = 1;
                if (usePrevResults)
                    initial =0;
                end
              else
                prevV = V(:,:,t-1);
                initial = 0;
              end
              d=diag(R).^(-1);
              Rinv=diag(d);
              quadTerm=C'*Rinv*C;
              if (measureTime)
                time(2)=time(2)+toc;
              tic
              end
              if (computeLogDet)
                    [V(:,:,t),K(:,:,t),Sinv(:,:,t),log_detS(:,t)] = ...
                            offline_kalman_update(A(:,:,m), C, Q, R, prevV,quadTerm,computeLogDet ,'initial', initial,'chol',1);
              else
                  [V(:,:,t),K(:,:,t),~,~] = ...
                            offline_kalman_update(A(:,:,m), C, Q, R, prevV,quadTerm,computeLogDet ,'initial', initial,'chol',1);
              end
              if (measureTime)
                time(1)=time(1)+toc;
              end

            end
            if (computeLogDet)
                varargout{1}=Sinv;
                varargout{2}=log_detS;
            end
            if (measureTime)
                keyboard
            end
        end
        
        function [Vnew,K,Sinv,log_detS] = Offline_update(A, C, Q, R, V,quadTerm,computeLogDet ,varargin)
        % KALMAN_UPDATE Do a one step update of the Kalman filter
        %
        % INPUTS:
        % A - the system matrix
        % C - the observation matrix 
        % Q - the system covariance 
        % R - the observation covariance
        % V(:,:) - Cov[X | y(:, 1:t-1)] prior covariance
        % quadTerm - quadratic term computed offline for efficency
        % computeLogDet: boolen that specifies whether we need to compute log_detS
        %           or not.
        %
        % OPTIONAL INPUTS (string/value pairs [default in brackets])
        % 'initial' - 1 means x and V are taken as initial conditions (so A and Q are ignored) [0]
        % 'chol' - boolean that specifies whether or not to use a cholesky
        % decomposition in computing log_detS. If stata space has high dimensionality
        % IT IS RECOMMENDED TO SET IT, due to better stability. 
        %
        % OUTPUTS (where X is the hidden state being estimated)
        %  Vnew(:,:) = Var[ X(t) | y(:, 1:t) ]
        %  K(:,:) = kalman gain at time t
        %  Sinv = Sinv(:,:,t) = inv(Covar( y(:,t)|y(:,1:t-1) )) 
        %  detS(:,t) = det(Covar( y(:,t)|y(:,1:t-1) ))


            % set default params
            initial = 0;
            useChol=0;

            args = varargin; %parse optional parameters
            for i=1:2:length(args)
              switch args{i}
               case 'initial', initial = args{i+1};
               case 'chol', useChol=args{i+1};
               otherwise, error(['unrecognized argument ' args{i}])
              end
            end

            measureTime=0;
            if(measureTime)
            time=zeros(1,7);    %1:A*mu & P, 
                                %2:CPC'+Sigma
                                %3:(CPC'+Sigma)^-1=Sinv
                                %4:log_detS
                                %5:K
                                %6:Vnew
            tic
            end
            if initial
              Vpred = V;
            else  
              Vpred = A*V*A' + Q;
              Vpred=(Vpred+Vpred')/2; %force symmetry (which might be lost due to roundings)
            end
            if(measureTime)
                time(1)=toc;
                tic
            end

            if not(useChol)
                S = C*Vpred*C' + R;
                S=(S+S')/2;
            end
            
            if(measureTime)
            time(2)=toc;
            tic
            end

            %EXPLOITING DIAGONAL COVAR MATRIX THNGS GET FASTER!
            d=diag(R).^(-1);
            Rinv=diag(d);
            %invV=inv(Vpred);
            invV=pinv(Vpred);
            Sinv=Rinv-(d*d').*(C/(invV+quadTerm)*C');
            Sinv=(Sinv+Sinv')/2;

            if (measureTime)
               time(3)=toc;  
               tic
            end
            
            ss = length(V);
            log_detS=NaN;
            if(computeLogDet)
                if (useChol==0)
                    determinant=abs(det(S));
                    log_detS=log(determinant);
                    if determinant==Inf
                        determinant1 = Inf;
                        regTerm=1;
                        while determinant1==Inf
                            regTerm=regTerm/2;
                            determinant1=abs(det(S*regTerm));
                        end
                        log_detS=log(determinant1)-length(S)*log(regTerm);
                        %from a simple tic/toc analysis, the method below to be waaay slower
                    %     E=eig(C);
                    %     logDet2=sum(log(E))
                    end
                    if determinant==0
                        determinant1 = 0;
                        regTerm=1;
                        while determinant1==0
                            regTerm=regTerm*2;
                            determinant1=abs(det(S*regTerm));
                        end
                        log_detS=log(determinant1)-length(S)*log(regTerm);
                    end
                else
                    log_detS=-Kalman.logdet(Sinv,'chol'); 
                end

            end

            if(abs(log_detS)==Inf || abs(log_detS)==-Inf)
                keyboard
            end

            if(measureTime)
                time(4)=toc;
                tic
            end
           
            K = Vpred*C'*Sinv; % Kalman gain matrix
            if(measureTime)
            time(5)=toc;
            tic
            end
            
            Vnew = (eye(ss) - K*C)*Vpred;
            Vnew=(Vnew+Vnew')/2;
            if(measureTime)
                time(6)=toc;
                tic
            end
            if (measureTime)
             keyboard
            end
        end
        
        function v = logdet(A, op)
        %LOGDET Computation of logarithm of determinant of a matrix
        %
        %   v = logdet(A);
        %       computes the logarithm of determinant of A. 
        %
        %       Here, A should be a square matrix of double or single class.
        %       If A is singular, it will returns -inf.
        %
        %       Theoretically, this function should be functionally 
        %       equivalent to log(det(A)). However, it avoids the 
        %       overflow/underflow problems that are likely to 
        %       happen when applying det to large matrices.
        %
        %       The key idea is based on the mathematical fact that
        %       the determinant of a triangular matrix equals the
        %       product of its diagonal elements. Hence, the matrix's
        %       log-determinant is equal to the sum of their logarithm
        %       values. By keeping all computations in log-scale, the
        %       problem of underflow/overflow caused by product of 
        %       many numbers can be effectively circumvented.
        %
        %       The implementation is based on LU factorization.
        %
        %   v = logdet(A, 'chol');
        %       If A is positive definite, you can tell the function 
        %       to use Cholesky factorization to accomplish the task 
        %       using this syntax, which is substantially more efficient
        %       for positive definite matrix. 
        %
        %   Remarks
        %   -------
        %       logarithm of determinant of a matrix widely occurs in the 
        %       context of multivariate statistics. The log-pdf, entropy, 
        %       and divergence of Gaussian distribution typically comprises 
        %       a term in form of log-determinant. This function might be 
        %       useful there, especially in a high-dimensional space.       
        %
        %       Theoretially, LU, QR can both do the job. However, LU 
        %       factorization is substantially faster. So, for generic
        %       matrix, LU factorization is adopted. 
        %
        %       For positive definite matrices, such as covariance matrices,
        %       Cholesky factorization is typically more efficient. And it
        %       is STRONGLY RECOMMENDED that you use the chol (2nd syntax above) 
        %       when you are sure that you are dealing with a positive definite
        %       matrix.
        %
        %   Examples
        %   --------
        %       % compute the log-determinant of a generic matrix
        %       A = rand(1000);
        %       v = logdet(A);
        %
        %       % compute the log-determinant of a positive-definite matrix
        %       A = rand(1000);
        %       C = A * A';     % this makes C positive definite
        %       v = logdet(C, 'chol');
        %

        %   Copyright 2008, Dahua Lin, MIT
        %   Email: dhlin@mit.edu
        %
        %   This file can be freely modified or distributed for any kind of 
        %   purposes.
        %

            %% argument checking
            assert(isfloat(A) && ndims(A) == 2 && size(A,1) == size(A,2), ...
                'logdet:invalidarg', ...
                'A should be a square matrix of double or single class.');
            if nargin < 2
                use_chol = 0;
            else
                assert(strcmpi(op, 'chol'), ...
                    'logdet:invalidarg', ...
                    'The second argument can only be a string ''chol'' if it is specified.');
                use_chol = 1;
            end

            %% computation
            if use_chol
                try
                     v = 2 * sum(log(diag(chol(A))));
                catch
                    disp('Matrix not Pos Def')
                        keyboard
                end
            else
                [L, U, P] = lu(A);
                du = diag(U);
                c = det(P) * prod(sign(du));
                v = log(c) + sum(log(abs(du)));
            end
            if (isnan(v)|| v==-Inf || v==Inf)
                keyboard
            end
        end


        function [J,V_smooth,VV_smooth]=Offline_smoother(A,Gamma,V_filt,varargin)

        % INPUTS:
        % A - the system matrix
        % Gamma - the process noise matrix
        % V_filt - filtered variances

        % OPTIONAL INPUTS (string/value pairs [default in brackets])
        % 'model' - allows for a time-varying model wherein A=A(t).  
        %     model(t)=m means use params from model m at time t [ones(1,T]
        %     In this case, matrix A takes an additional final dimension,
        %     i.e., A(:,:,m).
        % 'use_prev' - binary val. If equal to 1 exploits init_V in a different
        %       way, basically considering it as an actual estimate of the previous
        %       variance, not as a prior on initial process variance.
        %
        % OUTPUTS (where X is the hidden state being estimated)
        % V_smooth(:,:,t) = Cov[X(:,t) | y(:,1:T)]
        % VV_smooth(:,:,t) = E[X(:,t-1)x(:,t)' | y(:,1:T)], t>1
        % J - See Bishop 'Pattern recognition and machine learning' p. 641

            %% Hyperparameters and default initialization
            ss = size(A,1); % size of state space
            T=size(V_filt,3);
            % set default params
            model = ones(1,T);

            %% Parse optional arguments
            args = varargin;
            nargs = length(args);
            for i=1:2:nargs
              switch args{i}
               case 'model', model = args{i+1};
               otherwise, error(['unrecognized argument ' args{i}])
              end
            end
            
            %% Compute smoothed probabilities
            J=zeros(ss,ss,T);
            V_smooth=zeros(ss,ss,T);
            VV_smooth=zeros(ss,ss,T); % VV(:,:,t) = Cov[X(:,t), X(:,t-1) | y(:,1:T)] t >= 2
            V_smooth(:,:,T)=V_filt(:,:,T);
            for t=T-1:-1:1
               m = model(t+1);
               A_=A(:,:,m);
               P=A_*V_filt(:,:,t)*A_' +Gamma;
               J(:,:,t)=V_filt(:,:,t)*A_'/P;
               V_smooth(:,:,t)=V_filt(:,:,t)+J(:,:,t)*(V_smooth(:,:,t+1)-P )*J(:,:,t)';
               V_smooth(:,:,t)=(V_smooth(:,:,t)+V_smooth(:,:,t)')/2;
               VV_smooth(:,:,t+1)=V_smooth(:,:,t+1)'*J(:,:,t)';
               VV_smooth(:,:,t+1)=(VV_smooth(:,:,t+1)+VV_smooth(:,:,t+1)')/2;
            %    Vsmooth = Vfilt + J*(Vsmooth_future - Vpred)*J';
            end
        end

        
        function [xfilt] = Filter_Offline2Online_BATCH(y,x_init,K,A,C,varargin)
        % Once the observations data are available, performs kalman filtering based
        % on precomputed variances and gains
        %BATCH because we compute filtered states for each I in one go,
        %       which is decisively faster

        % INPUT:
        % y - observations (frames of videos)
        % x_init -expected value of latent variabvle at time 0
        % K - dimState x dimObservations x T: kalman gains, one for every timestep.
        % A - state evolution matrix
        % C - observation matrix
        
        % OPTIONAL INPUTS (string/value pairs [default in brackets])
        % 'model' - allows for a time-varying model wherein A=A(t).  
        %     model(t)=m means use params from model m at time t [ones(1,T]
        %     In this case, matrix A takes an additional final dimension,
        %     i.e., A(:,:,m).
        
        % OUTPUT:
        % xfilt: filtered state estimates

            %% Hyperparameters and default values
            [f,I,T]=size(y);
            ss = size(A,1);
            model = ones(1,T);

            
            args = varargin;
            nargs = length(args);
            for i=1:2:nargs
              switch args{i}
               case 'model', model = args{i+1};
               otherwise, error(['unrecognized argument ' args{i}])
              end
            end
            
            %% Filtering based on observations (hence the name 'online')
             % and kalman gain matrix
            CA=zeros(f,ss,size(A,3));   %CA saves us some computations 
                                        %if size(unique(model))<<T
            for k=unique(model)
               CA(:,:,k)=C*A(:,:,k); 
            end

            xfilt = zeros(ss,I, T);
            xfilt(:,:,1)=repmat(x_init,1,I)+K(:,:,1)*(y(:,:,1)-C*repmat(x_init,1,I));

            for t=2:T
                m = model(t);
                xfilt(:,:,t)=A(:,:,m)*xfilt(:,:,t-1)+K(:,:,t)*(y(:,:,t)-CA(:,:,m)*xfilt(:,:,t-1));
            end

        end
        
        
        function E_z_smooth=smoother_Offline2Online_BATCH(A,J,E_z_filt,varargin)
        % Once sobservations are available, computes smoothed expected
        % values, based on filtered expected values and offline results
        % BATCH because we compute filtered states for each I in one go,
        %       which is decisively faster
        %INPUT:
        % A - state matrix
        % J - See Bishop 'Pattern recognition and machine learning' p. 641
        % E_z_filt(:,i,t) expected valued of latent variable for t-th frame of i-th
        %            video. 
        %OPTIONAL INPUT:
        % model: allows for a time-varying model wherein A=A(t). 
        %
        %OUTPUT: 
        % E_z_smooth(:,i,t) expected value for x(i,t), the latent variable
        %of the i-th video at time T. 
        
            %% Hyperparameters and default value for model
            [ss,I,T]=size(E_z_filt);
            model = ones(1,T);

            %% parse optional input
            args = varargin;
            nargs = length(args);
            for i=1:2:nargs
              switch args{i}
               case 'model', model = args{i+1};
               otherwise, error(['unrecognized argument ' args{i}])
              end
            end

            %% perform batch smoothing
            E_z_smooth=zeros(ss,I,T);
            E_z_smooth(:,:,T)=E_z_filt(:,:,T);

            for t=T-1:-1:1
                m=model(t+1);
                E_z_smooth(:,:,t)=E_z_filt(:,:,t)+J(:,:,t)*(E_z_smooth(:,:,t+1)-A(:,:,m)*E_z_filt(:,:,t));
            end

        end

        function [loglik]=Loglik_Offline2Online_BATCH(C,A,Sinv,log_detS,initState,x_filt,y,varargin)
        % Once the observations data are available, computes the likelihood of the observations, based on precomputed 
        % filtered expected values and innovation covariance matrices

        % OUTPUT:
        % loglik: log-likelihood of observed sequence

        % INPUT:
        % y - observations (frames of videos)
        % x_filt -expected value of latent variables (computed with KF)
        % S_inv - inverse of the innovation covariance matrix
        % log_detS - logarithm of determinant of innovation covariance matrix
        % K - dimState x dimObservations x T: kalman gains, one for every timestep.
        % A - state evolution matrix
        % C - observation matrix

        % OPTIONAL INPUT:
        % model: allows for a time-varying model wherein A=A(t). 

            %% Hyperparameters, time for tic-toc, default model value
            %y: f x I x T
            %x_filt: Dh+Dw x I xT
            [f,I,T]=size(y);
            ss=size(A,1);
            takeTime=0;

            if (takeTime)
                time=zeros(1,2); % 1 - y-CA*x_filt
                                 % 2 - anything else
            end
            model = ones(1,T);

            %% parse optional arguments
            args = varargin;
            nargs = length(args);
            for i=1:2:nargs
              switch args{i}
               case 'model', model = args{i+1};
               otherwise, error(['unrecognized argument ' args{i}])
              end
            end

            %% actual loglikelihood math
            CA=zeros(f,ss,size(A,3));
            for k=unique(model)
                CA(:,:,k)=C*A(:,:,k);
            end

            if (takeTime)
                tic
            end
            
            temp=y(:,:,1)-repmat(C*initState,1,I);
            mahal=sum(temp.*(Sinv(:,:,1)*temp));
            if (takeTime)
                time(2)=toc;
                tic
            end
            loglik=-0.5*mahal - (f/2)*log(2*pi)-log_detS(1)/2;
            for t=2:T
                m=model(t);
                if (takeTime)
                    tic
                end
                temp=(y(:,:,t)-CA(:,:,m)*x_filt(:,:,t-1));
                if (takeTime)
                    time(1)=time(1)+toc;
                    tic
                end
                mahal=sum(temp.*(Sinv(:,:,t)*temp));
                if (takeTime)
                    tic
                    time(2)=time(2)+toc;
                    tic
                end
                loglik=loglik-0.5*mahal - (f/2)*log(2*pi)-log_detS(t)/2;
            end

            if(any(isnan(loglik)) || any(loglik==-Inf) || any(loglik==Inf))
                keyboard
            end

            if (takeTime)
                keyboard
            end

        end

        
        
    end

end