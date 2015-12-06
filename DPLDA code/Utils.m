classdef Utils
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % generic utility functions shared throughout the code
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    methods(Static)
        
        function [f,I,T]=getSizesFrom(X_all)
        % INPUT:
        %   X_all{J} contains all the people with exactly J videos. X_all{J}(:,i,j,t) is the
        %               t-th frame of the j-th video of the i-th person (among the
        %               ppl depicted in exactly J videos).
        % OUTPUT
        %   f: number of festures.
        %   I: number of different individuals
        %   T: number of frames per video
            I=0;
            f=0;
            T=0;
           for ind=1: size(X_all,2)
               x=X_all{ind};
               if not(size(x)==0)
                  I=I+size(x,2);
                  f=size(x,1); 
                  T=size(x,4);
               end
           end
           
        end
        
        function [N]=totNumVids(X_all)
            N=0;
            numBucketsTrain=length(X_all);
            nonEmptyBucketsTrain=[];
            for ind=1:numBucketsTrain
                if not(isempty(X_all{ind}))
                    nonEmptyBucketsTrain=[nonEmptyBucketsTrain,ind];
                end
            end
           for b=nonEmptyBucketsTrain
           I=size(X_all{b},2);
           N=N+I*b;
           end
           
        end    
        
        function [A_stack,C_stack,Sigma_stack,Gamma_stack,mu_stack] = stackMatrices(F,G,A,Sigma,mu,State_Augment,T,Dh,Dw)
        % OUTPUT: 4 'global' matrices describing the dynamics of the
        % augmented, (Dh+State_Augment*Dw)-size, system
        % INPUT:  F h-to-x matrix
        %         G w-to-x matrix
        %         A state evolution matrix
        %         Sigma observation noise matrix
        %         J: number of videos per person
        %         State_augment: how many vectors are we stacking up?
        %               normally State_augment=J, but there can be
        %               exceptions
        %         Aij = 1 if different state evlution matrix per video
        %              ONLY WORKS WITH KALMAN LONG, THE OTHER 2 METHODS ARE
        %              OUTDATED

            Gdiag=G;
            for i=1:State_Augment-1
               Gdiag=blkdiag(Gdiag,G); 
            end
            C_stack=[repmat(F,State_Augment,1),Gdiag]; %size: State_Augment*fx(JDw+Dh)

            A_stack=eye(Dh);
            for i = 1:State_Augment
                A_stack=blkdiag(A_stack,A); %final size: (Dh+State_Augment*Dw)x(Dh+State_Augment*Dw)
            end

            Sigma_stack=Sigma;
            for i=1:State_Augment-1
                Sigma_stack=blkdiag(Sigma_stack,Sigma);%final size: State_Augment*fxState_Augment*f
            end
            
            Gamma_stack=zeros(Dh);
            for i=1:State_Augment
                Gamma_stack=blkdiag(Gamma_stack,eye(Dw));% final size: (Dh+State_Augment*Dw)x(Dh+State_Augment*Dw)
            end
            
            mu_stack=repmat(mu,State_Augment,T);
        
        end
        
        function[norms]=getNorm(varargin)
            norms=zeros(1,nargin);
           for i=1:nargin
               norms(i)=norm(cell2mat(varargin(i)));
           end
        end
        
        function[dist]=getDistance(matr1,matr2)
           d1=max(max(abs(matr1-matr2)./abs(matr2)));
           d2=max(max(abs(matr1-matr2)./abs(matr1)));
           dist= max(d1,d2);
        end
        
        function [converged, decrease] = em_converged(loglik, previous_loglik, threshold, check_increased)
        % Implemented by Kevin Murphy
        % EM_CONVERGED Has EM converged?
        % [converged, decrease] = em_converged(loglik, previous_loglik, threshold)
        %
        % We have converged if the slope of the log-likelihood function falls below 'threshold', 
        % i.e., |f(t) - f(t-1)| / avg < threshold,
        % where avg = (|f(t)| + |f(t-1)|)/2 and f(t) is log lik at iteration t.
        % 'threshold' defaults to 1e-4.
        %
        % This stopping criterion is from Numerical Recipes in C p423
        %
        % If we are doing MAP estimation (using priors), the likelihood can decrase,
        % even though the mode of the posterior is increasing.

        if nargin < 3, threshold = 1e-4; end
        if nargin < 4, check_increased = 1; end

        converged = 0;
        decrease = 0;

        if check_increased
          if loglik - previous_loglik < -1e-3 % allow for a little imprecision
            fprintf(1, '******likelihood decreased from %6.4f to %6.4f!\n', previous_loglik, loglik);
            decrease = 1;
        converged = 0;
        return;
          end
        end

        delta_loglik = abs(loglik - previous_loglik);
        avg_loglik = (abs(loglik) + abs(previous_loglik) + eps)/2;
        if (delta_loglik / avg_loglik) < threshold, converged = 1; 
        end
        delta_loglik / avg_loglik;
        end
        
        
        function [X_gal, X_probe,X_gal_IDs,X_probe_IDs] = getGalleryProbeFrom(X_all,X_all_IDs,j_probe)
        % this function devides data into gallery (possibly also used for training) 
        % and probe
        % INPUT:
        %   X_all{J} contains all the people with exactly J videos. X_all{J}(:,i,j,t) is the
        %               t-th frame of the j-th video of the i-th person (amog the
        %               ppl depicted in exactly J videos)
        %   X_all_IDs{J} contains the numeric IDs of the ppl with exctly J videos.
        %               X_all_IDs{j}(i) contains the ID of the i-th person
        %   j_probe deterines which video will be used as a probe for all the subjects.

        % OUTPUT
        %   X_gal: gallery data, with structure analogous to X_all
        %   X_gal_IDs : IDs with the same structure as above
        %   X_probe: probe videos of size numFeatures x numPeople x T
        %   X_probe_IDs : ID of the probe videos
        
            args = varargin; %parse optional arguments
            nargs = length(args);
            Num_ext=0;
            X_ext=0;
            
            for i=1:2:nargs
              switch args{i}
                  case 'X_ext', X_ext = args{i+1};
                  case 'Num_ext', Num_ext=args{i+1};
               otherwise, error(['unrecognized argument ' args{i}])
              end
            end

            [f,totI_int,T]=Utils.getSizesFrom(X_all);
            totI=totI_int+Num_ext;
            X_probe=zeros(f,totI,T);
            X_probe_IDs=zeros(totI,1);
            X_gal=cell(1,size(X_all,2)-1);
            X_gal_IDs=cell(1,size(X_all,2)-1);
            count=1;

            for ind=1:size(X_all,2)
                J_all=1:ind;
                J_train=J_all(not(J_all==j_probe));
               vids=X_all{ind};
               for ind_i=1:size(vids,2)
                   X_probe(:,count,:)=vids(:,ind_i,j_probe,:);
                   X_probe_IDs(count)=X_all_IDs{ind}(ind_i);
                   count=count+1;
               end
               if (not(isempty(vids)) && ind>1)
                   X_gal{ind-1}=vids(:,:,J_train,:); 
                   X_gal_IDs{ind-1}=X_all_IDs{ind};
               end

            end
            if(Num_ext>0)
                if (not(X_ext==0))
                    indStartExt=(j_probe-1)*Num_ext+1;
                    X_probe(:,count:end,:)=X_ext(:,indStartExt+(1:Num_ext),:);
                    X_probe_IDs(count:end)=totI_int+1;
                else
                    disp('X_ext is zero, Num_ext is larger than zero, you re doing OS_identification wrong')
                end
            end
        end
        
                
        
        function [ ImagesPLDA , ImageID ] = PreprocessData4PLDA( X_train,X_train_IDs,N,rand )
        % Given video data in a format convenient for DPLDA converts into a
        % image-set representation, convenient for PLDA functions.

        %       INPUT
        %       X_train: gallery images
        %       X_train_IDs: identities associated to images in consistent format
        %       N: number of images we want to use per video
        %       rand: if (rand) we pick N videos at random, if !(rand) we pick the
        %       first N.
        %
        %       OUTPUT
        %       ImagesPLDA: NFeature x NSample  -  Training data
        %       ImageID: NSample x nIdentity  -  Identity matrix of training data

        %X_train=zeros(f,I,J,T);
        numBucketsTrain=length(X_train);
        nonEmptyBucketsTrain=[];
        for ind=1:numBucketsTrain
            if not(isempty(X_train{ind}))
                nonEmptyBucketsTrain=[nonEmptyBucketsTrain,ind];
            end
        end

        [f,numIdents,T]=Utils.getSizesFrom(X_train);
        numVid=Utils.totNumVids(X_train);


        ImagesPLDA=zeros(f,numVid*N);
        rows=[]; %sample
        cols=[]; %identity


        count=1;
        for b=nonEmptyBucketsTrain
            I=size(X_train{b},2);
            for i=1:I   
               for j=1:b
                   randIndices=(1:N);
                   if rand
                       perm =randperm(T);
                       randIndices=perm(1:N);
                   end
                  for t=randIndices
                      ImagesPLDA(:,count)=X_train{b}(:,i,j,t);
                      rows=[rows,count];
                      ident=X_train_IDs{b}(i);
                      cols=[cols,ident];          
                      count=count+1;
                  end
               end    
            end
        end

        ImageID=sparse(rows,cols,ones(1,numVid*N),numVid*N,numIdents);
        end
        
    end

end