%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% DPLDA closed set identification
%
% Selecting different values for Dh (size of shared space), Dw (size of
% private space), j_probe (index of video to be used as probe) and L
% (number of iterations of EM algorithm), this script tries all possible
% combinations in performing Closed set identification, and stores the
% successRate for each of these combinations in a conveniently sized array.

% The data, called X_train and loaded from elsewhere, is in the format X_train=cell{1,max_J}, 
% where max_J is the max number of videos of the same person. Every person with j
% videos is saved inside X{j}. In particular X_train{j}=double(numFeatures,numPeople,j,numFrames)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


clear

Dh_=[50]; %size of identity subspace
Dw_=[2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50]; %size of accidental conditios subspace
L_=[4]; %max EM iterations
conv_thresh=10e-3; %convergence threshold for EM algorithm
j_probe_=[1,2,3,4,5]; %index of video to be used as probe


successRate=zeros(length(Dh_),length(Dw_),length(L_),length(j_probe_));

load '../../../../../../vol/atlas/homes/alessandro/DataTrain/Identification/50_faces_all_f=7930_avgVids=0_T=40.mat'


for index1=1:length(Dh_)
    for index2=1:length(Dw_)
        for index3=1:length(L_)
            for index4=1:length(j_probe_) %this guy will go eventually
%% load data

% load C:\Users\sandr_000\Desktop\Todo\UNI\Imperial\Tesi\MyCode\provaSeed.mat
j_probe=j_probe_(index4);
Dh=Dh_(index1);
Dw=Dw_(index2);
L=L_(index3);
[Dh,Dw,j_probe,L]

[X_gal, X_probe,X_gal_IDs,X_probe_IDs]=Utils.get_gallery_probe_from(X_train,X_train_IDs,j_probe);
[f,~,T_train]=Utils.get_sizes_from(X_gal);
numBucketsTrain=length(X_gal);
nonEmptyBucketsTrain=[];
for ind=1:numBucketsTrain
    if not(isempty(X_gal{ind}))
        nonEmptyBucketsTrain=[nonEmptyBucketsTrain,ind];
    end
end


%%
count_tr=0;
sum=zeros(f,1);
for b=nonEmptyBucketsTrain
    temp=X_gal{b};
    for i=1:size(temp,2)
        for j=1:b
            for t=1:T_train
               sum=sum+ temp(:,i,j,t);
               count_tr=count_tr+1;
            end
        end
    end
end
mu_init=sum/count_tr;

%% initialize at random
% seed=rng;
% %rng(seed)
% F_init=rand(f,Dh);
% G_init=rand(f,Dw);
% Sigma_init=abs(diag(rand(1,f)));
% % eig_=2*rand(1,Dw)-1;
% eig_=rand(1,Dw);
% BM=orth(rand(Dw));
% A_init=BM*diag(eig_)*BM';

%% initialize with PLDA

addpath('../../PLDA modified');
N=T_train;
[X_PLDA_train, identities_PLDA_train]=Utils.preprocess_data_4PLDA(X_gal,X_gal_IDs,N,0);
Model_PLDA=PLDA_Train(X_PLDA_train, identities_PLDA_train, L, Dh, Dw,0,0,0);
F_init=Model_PLDA.F;
G_init=Model_PLDA.G;
Sigma_init=diag(Model_PLDA.Sigma);
% eig_=2*rand(1,Dw)-1;
eig_=rand(1,Dw);
BM=orth(rand(Dw));
A_init=BM*diag(eig_)*BM';


foldSave=strcat('../../../../../../vol/atlas/homes/alessandro/Results/CS_ident/Model/PLDA/',num2str(j_probe),'/');
mkdir(foldSave);
fileSave=strcat(foldSave,'Mod_50faces_noPCA_T=40_Dh=',num2str(Dh),'_Dw=',num2str(Dw))
save(fileSave,'Model_PLDA');

%% EM algorithm for parameter estimate
[ A_est,F_est,G_est,Sigma_est,mu_est, progress_lik ] = EM_estimate(A_init, F_init, G_init, Sigma_init, mu_init,...
    X_gal,T_train,L,conv_thresh,f,Dh,Dw,0);

% plot(progress_lik);
%% Identification of probe videos
% real identities (ground truth) are inside X_probe_IDs
[f,~,T_gal]=Utils.get_sizes_from(X_gal);
[ conditionalLogLiks, identities ] = identification( A_est,F_est,G_est,Sigma_est,mu_est, X_gal,X_gal_IDs , X_probe,T_gal,Dh,Dw,0);
successfulIdent=find(X_probe_IDs==identities');
successRate(index1,index2,index3,j_probe)=length(successfulIdent)/length(identities);

            end
        end
    end    
end

%% Store success rate

pathSave='../../../../../../vol/atlas/homes/alessandro/Results/CS_ident/';
mkdir(pathSave);
string = strcat (pathSave,'50faces_PCA_whitened_f=',num2str(f),'_T=',num2str(T_gal),'.mat')
save(string,'successRate')



