function [Rtotal,Rarr,simuProjs] = calc_Rfactor_realspace_general(model,Projs,Angles,vector1,verctor2,vector3)
% % This function is just bridging the backProj_Rfactor, which pass the
% Projs and Angles as array to Tian_calc_Rfactor_realspace; while
% Tian_calc_Rfactor_realspace will pass individual proj and angles to
% calculate3Dprojections_RealSpaceinterpfunction.

% initialize error array Rarr (version1 uses horizontal angle array)
Rarr = zeros(1,size(Angles,1)); %% So here we are using the horizontal Angles.

% initialize back projection stack
simuProjs = zeros(size(Projs,1),size(Projs,2));

for i=1:size(Angles,1)
    currProj = Projs(:,:,i);
    backProjs = calculate3Dprojection_RealSpaceinterp_general(model,Angles(i,1),Angles(i,2),Angles(i,3),vector1,verctor2,vector3);
    backProjs = My_stripzero(backProjs,size(currProj));
    Rarr(i) = sum( abs(abs(backProjs(:))-abs(currProj(:)))) / sum(abs(currProj(:)));
    fprintf('Projection %g with Rfactor: %.5g \n',i,Rarr(i));
    simuProjs = cat(3,simuProjs,backProjs);
end
simuProjs = single(simuProjs);
simuProjs(:,:,1)=[];
Rtotal = mean(Rarr);
end
