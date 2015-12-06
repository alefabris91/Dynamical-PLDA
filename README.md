# Dynamical-PLDA
Open and closed set face identification with DPLDA

DPLDA (Dynamical Probabilistic Linear Discriminant Analysis) is the first dynamical generalization of PLDA. 
It makes PLDA more efficient and available for data that inherently comprises temporal information, 
such as videos and inertial measurements sequences. 

DPLDA code contains the fundamental routines to run DPLDA models. 
  Start from func_CS_Identification_DPLDA.m, func_OS_Identification_DPLDA. 
  They allow to exploit DPLDA models for applications of face identification, both closed-set and open-set.

  If you want to delve deeper into the model's internal functioning,
  EM_estimateLong_OL_parfor.m and identifyLongKalman_MO_parfor.m deal with model identification and inference.

  Kalman.m contains routines related to Kalman filtering.

  Utils.m is populated with generic utility (not particularly interesting) funcitons.
  
PLDA code, LDA code, PCA code contain matlab code, developed by third parties.
  We reccomend downloading all of the above for two reasons:
  - some functions within PLDA code can be used to initialize DPLDA models.
  - they can be used to directly compare DPLDA models's performance with
  the one achieavable through PLDA, LDA, PCA models. 



