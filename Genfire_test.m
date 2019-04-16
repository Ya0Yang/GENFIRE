t1 = clock;

pj_filename         = 'data_turb3/pj_new_Turbu3_1015.mat';
angle_filename      = 'data_turb3/angles_usedin_refine_1015.mat';

results_filename    = 'results/Genfire_tomo_res.mat';
filenameFinalRecon  = 'results/Genfire_tomo_recon.mat';

%% GENFIRE parameters

doGPU = 1;

GENFIRE = GENFIRE_Reconstructor();

%%% See the README for description of parameters

GENFIRE.filename_Projections = pj_filename;
GENFIRE.filename_Angles = angle_filename ;
GENFIRE.filename_Results = results_filename;

GENFIRE.oversamplingRatio = 4;
GENFIRE.numIterations = 200; 
GENFIRE.interpolationCutoffDistance =.125; 

% GENFIRE.ds_value = 0.85;
GENFIRE.griddingMethod = 2; 

GENFIRE.allowMultipleGridMatches = 1;
GENFIRE.constraintEnforcementMode = 3; 
GENFIRE.constraintPositivity = 1;
GENFIRE.constraintSupport = 1;

GENFIRE.DFT_doGPU = doGPU;
%GENFIRE.vector1 = [0 0 1];
%GENFIRE.vector2 = [0 1 0];
GENFIRE.vector3 = [1 0 0];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Begin GENFIRE

GENFIRE = readFiles(GENFIRE);
GENFIRE = CheckPrepareData(GENFIRE);
GENFIRE = runGridding(GENFIRE); 
GENFIRE = reconstruct(GENFIRE);
% SaveResults(GENFIRE);
%%

[Rfactor,Rarray,simuProjs]=Tian_calc_Rfactor_realspace...
    (GENFIRE.reconstruction,GENFIRE.InputProjections,GENFIRE.InputAngles);
final_Rec = GENFIRE.reconstruction;
final_errK = GENFIRE.errK;
final_Rfactor = Rfactor;
final_Kfactor = min(final_errK);
final_Rarray = Rarray;
final_simuProjs = simuProjs;

% GENFIRE = ClearCalcVariables(GENFIRE);

save('Factor_GENFIRE_test_gdref.mat', 'final_Rec','final_errK','final_Rfactor','final_Kfactor','final_Rarray','final_simuProjs');
% save(results_filename, 'GENFIRE','-v7.3');
t2 = clock;
fprintf('Completed in %.02f seconds\n',etime(t2,t1))

