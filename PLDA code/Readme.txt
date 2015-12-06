Matlab implementation of Probabilistic Linear Discriminant Analyzer (PLDA)
%******************** Disclaimer *****************************************
% This program is distributed in the hope that it will be useful, but
% WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
%
% The program is free for academic use. If you are interested in using the
% software for commercial purposes.  Please contact 
%   Dr. Simon J. D. Prince  
%   Email: s.prince@cs.ucl.ac.uk  
% 
% Plase use the citation provided below if it is useful to your research:
%
% S.J.D. Prince and J.H. Elder, “Probabilistic linear discriminant analysis
% for inferences about identity,”  ICCV, 2007. 
%
% P. Li and S.J.D. Prince, “Probabilistic Methods for Face Registration and
% Recognition”, In Advances in Face Image Analysis: Techniques and
% Technologies, Y. Zhang (eds.)  (in press). 
% 
%**************************************************************************
% 04-03-2010

% Main subrotine for PLDA training and test
PLDA_Train.m 		- Training PLDA model using EM algorithm
PLDA_Identification.m 	- Face identification test using learned PLDA model 
PLDA_verification.m 	- Face verification test for two images using learned PLDA model

% Demo 
PLDA_Train_Demo.m           - Example of training a PLDA model
PLDA_Identification_Demo.m 	- Example of face identification test
PLDA_Identification_Demo2.m - Example of face identification test using 
                              trained PLDA models with different subspace
                              dimension. This produces the curve in the 
                              book chapter.

PLDA_Verification_Demo.m 	- Example of face verification test
Rand_Feature_Sampling_Demo  - Example of face identification using an ensemble
                              of PLDA models trained by random subset of raw 
                              pixel intensities of the face image.

% Trained PLDA model Demo with 7 different subspace dimensions. This produces 
the curve in the book chapter. 

PLDAModels.mat      -   7 PLDA models trained using the following XM2VTS data subset

% Data
Data.mat - Concatenated pixel (RGB) data subset from XM2VTS database for illustration. 
This data subset is used in the two papers aforementioned. It consists of 
the following variables:
    DataTrain   - Training data (14700 x 603 matrix), with concatenated 
                  pixel (RGB) data of 603 images from the first 195 
                  identities. The size of each image is of 70 x 70. 
                  So the dimenion of feature is 70 x 70 x 3 = 14700.

    ImageID     - Identity matrix (603 x 195). Each column is an individual
                  and each row is an image. An element in the matrix with 
                  value 1 means the image of this row is from identity of 
                  this column, 0 otherwise.

    DataGallery - Gallery image data (14700 x 100 matrix) of concatenated 
                  pixel data of 100 images from the last 100 identities in 
                  XM2VTS database. 

    DataProbe   - Probe image data (14700 x 100 matrix) of concatenated 
                  pixel data of 100 images from the last 100 identities in 
                  XM2VTS database. The orders of the identities in the 
                  gallery and probe data are the same.


