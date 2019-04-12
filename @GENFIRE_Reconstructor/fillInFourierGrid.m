%%  fillInFourierGrid %%

%%inputs:
%%  projections - measured projections
%%  angles - Euler angles in the form 3xN_projections, where each projection has 3 angles in the form [phi;theta;psi]
%%  interpolationCutoffDistance - radius of sphere in which to include measured
%%  oversamplingRatio - oversampling ratio for the projections in each direction
%%      values when filling in grid. All points within this sphere will be weighted
%%      linearly by their inverse distance. 
%%  interpolationCutoffDistance - radius of interpolation kernel
%%  doCTFcorrection - flag to correct for Contrast Transfer Function (CTF) in projections, requires CTFparameters
%%  CTFparameters - structure containing defocus values and defocus angle for each projection
%%  allowMultipleGridMatches - whether or not to allow each measured datapoint to be matched to multiple grid points

%%outputs:
%%  rec - inverse FFT of the assembled Fourier grid
%%  measuredK -assembled Fourier Grid

%% Author: Alan (AJ) Pryor, Jr.
%% Jianwei (John) Miao Coherent Imaging Group
%% University of California, Los Angeles
%% Copyright (c) 2015. All Rights Reserved.




function obj = fillInFourierGrid(obj)

projections = obj.InputProjections;
angles = obj.InputAngles;
particleWindowSize = obj.particleWindowSize(2);
interpolationCutoffDistance = obj.interpolationCutoffDistance;
allowMultipleGridMatches = obj.allowMultipleGridMatches;
oversamplingRatio = obj.oversamplingRatio;
%,particleWindowSize,oversamplingRatio,interpolationCutoffDistance,doCTFcorrection, [], allowMultipleGridMatches

%initialize array to hold measured data
if mod(particleWindowSize,2)==0
    kMeasured = zeros(particleWindowSize*oversamplingRatio,particleWindowSize*oversamplingRatio,size(projections,3));
else
    kMeasured = zeros(1 + (particleWindowSize-1)*oversamplingRatio,1 + (particleWindowSize-1)*oversamplingRatio,size(projections,3));  
end

tic %start clock

%get the dimension (assumed square and even) and setup the center and radius of the array size
dim1 = size(kMeasured,1);
nc = single(round((dim1+1)/2));%center pixel
n2 = single(nc-1);%radius of array

%setup the coordinates of the reciprocal slice to determine its 3D coordinates
[ky, kx] = meshgrid((1:dim1)-nc,(1:dim1)-nc);ky = single(ky);kx = single(kx);
kx = single(kx(:))'; ky = single(ky(:))'; %initialize coordinates of unrotate projection slice
kz = zeros(1,dim1*dim1,'single'); %0 degree rotation is a projection onto the X-Y plane, so all points have kz=0;

%check for the presence of some of the CTF correction options and set defaults if they are absent

%otherwise, add the projection to the stack of data with no further corrections
for projNum = 1:size(projections,3)
    kMeasured(:,:,projNum) = my_fft(My_paddzero(projections(:,:,projNum),[size(kMeasured,1) size(kMeasured,2)]));
end

clear projections

%initialize arrays to contain coordinates
measuredX = zeros(1,size(kMeasured,2)*size(kMeasured,1),size(kMeasured,3),'single');
measuredY = zeros(1,size(kMeasured,2)*size(kMeasured,1),size(kMeasured,3),'single');
measuredZ = zeros(1,size(kMeasured,2)*size(kMeasured,1),size(kMeasured,3),'single');


for projNum = 1:size(kMeasured,3)
phi = angles(projNum,1);
theta = angles(projNum,2);
psi = angles(projNum,3);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  GENFIRE/RELION/XMIPP/FREALIGN/EMAN Euler angle convention:
% % 
R = [ cosd(psi)*cosd(theta)*cosd(phi)-sind(psi)*sind(phi) ,cosd(psi)*cosd(theta)*sind(phi)+sind(psi)*cosd(phi)   ,    -cosd(psi)*sind(theta);
      -sind(psi)*cosd(theta)*cosd(phi)-cosd(psi)*sind(phi), -sind(psi)*cosd(theta)*sind(phi)+cosd(psi)*cosd(phi) ,   sind(psi)*sind(theta)  ;
      sind(theta)*cosd(phi)                               , sind(theta)*sind(phi)                                ,              cosd(theta)];

rotkCoords = R'*[kx;ky;kz];%rotate coordinates
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

measuredX(:,:,projNum) = rotkCoords(1,:);%rotated X
measuredY(:,:,projNum) = rotkCoords(2,:);%rotated Y
measuredZ(:,:,projNum) = rotkCoords(3,:);%rotated Z
end

%reshape to simplify
measuredX = reshape(measuredX,1,size(kMeasured,2)*size(kMeasured,1)*size(kMeasured,3));
measuredY = reshape(measuredY,1,size(kMeasured,2)*size(kMeasured,1)*size(kMeasured,3));
measuredZ = reshape(measuredZ,1,size(kMeasured,2)*size(kMeasured,1)*size(kMeasured,3));
kMeasured = reshape(kMeasured,1,size(kMeasured,2)*size(kMeasured,1)*size(kMeasured,3));
badInd = find(kMeasured==-999);%delete values that are flagged as bad
measuredX(badInd) = [];
measuredY(badInd) = [];
measuredZ(badInd) = [];
kMeasured(badInd) = [];

masterInd = [];%masterInd will be a large list of the grid indices
masterVals = [];%complex values to include in weighted averaging for those grid points
masterDistances = [];%distance from measured value to grid point
% masterConfidenceWeights = [];

if allowMultipleGridMatches
    shiftMax = round(interpolationCutoffDistance);
else
    shiftMax = 0;
end
%The nearest grid point to a measured value can be found by rounding, but
%there can be more than one grid point within the cutoff sphere, so must
%search locally for other possibilities. However in practice I have found 
%this search can slow the program down greatly, without significant change
%in final result. Even searching 1 voxel in either direction increases the
%number of calculations by 3^3 = 27; For this reason I have set shiftMax = 0 and
%just assign values to their closest voxel.

for Yshift = -shiftMax:shiftMax 
   for Xshift = -shiftMax:shiftMax
       for Zshift = -shiftMax:shiftMax
            tmpX = (round(measuredX)+Xshift); % apply shift
            tmpY = (round(measuredY)+Yshift);
            tmpZ = (round(measuredZ)+Zshift);
            tmpVals = kMeasured;
            distances = sqrt(abs(measuredX-tmpX).^2+abs(measuredY-tmpY).^2+abs(measuredZ-tmpZ).^2); %compute distance to nearest voxel
            tmpY = tmpY+nc; %shift origin
            tmpZ = tmpZ+nc;
            tmpX = tmpX+nc;
            goodInd = (~(tmpX>dim1|tmpX<1|tmpY>dim1|tmpY<1|tmpZ>dim1|tmpZ<1)) & distances<=interpolationCutoffDistance;%find candidate values
            masterInd = [masterInd sub2ind([dim1 dim1 dim1],tmpX(goodInd),tmpY(goodInd),tmpZ(goodInd))]; %append values to lists
            masterVals = [masterVals tmpVals(goodInd)];
            masterDistances = [masterDistances distances(goodInd)];

       end
   end
end
   
clear measuredX
clear measuredY
clear measuredZ
clear confidenceWeights

% Now that we have a list of the complex values to grid, their coordinates, 
% and their distances from the nearest voxel, we want to reorganize the
% data so that all values matched to a given voxel are in the same place,
% so that the weighted sum can be computed. The number of values matched to
% each voxel can vary, and although one could use cell arrays for this
% purpose, they are quite slow. Instead, one can simply sort the indices,
% and then find the unique values by looking at the difference in
% consecutive elements. 

masterDistances = masterDistances + 1e-5;
masterDistances(masterDistances>0) = 1 ./ masterDistances(masterDistances>0);
masterDistances(isnan(masterDistances)) = 0;

obj.measuredK = accumarray(masterInd',masterVals.*masterDistances,[dim1^3 1]);
sumWeights = accumarray(masterInd',masterDistances,[dim1^3 1]);
obj.measuredK(sumWeights>0) = obj.measuredK(sumWeights>0) ./ sumWeights(sumWeights>0);
obj.measuredK = reshape(obj.measuredK,[dim1 dim1 dim1]);
obj.measuredK = hermitianSymmetrize(obj.measuredK);

obj.recIFFT = My_stripzero(real(my_ifft(obj.measuredK)), [obj.Dim1 obj.Dim2 obj.Dim1]);
timeTakenToFillInGrid = toc;
timeTakenToFillInGrid = round(10*timeTakenToFillInGrid)./10;
fprintf('GENFIRE: Fourier grid assembled in %.12g seconds.\n\n',timeTakenToFillInGrid);
