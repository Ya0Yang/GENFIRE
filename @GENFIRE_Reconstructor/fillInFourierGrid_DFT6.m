% fillInFourierGrid_DFT_Class
% Y. Yang, UCLA Physics & Astronomy
% First version date: 2015. 04. 30.
% output parameter: rec (nx x ny x nx) IFFT of ftArray
%                   ftArray (nx x ny x nx) interpolated Fourier 3D array
%                   CWArray (nx x ny x nx) interpolated confidence weight
%                   SDArray (nx x ny x nx) interpolated distance
%
%
% input parameter: obj.InputAngles - measured projection, (nx x ny x P) array with P # of projections
%                  obj.InputAngles - 3 Euler angle array, (3 x P) array
%                  obj.interpolationCutoffDistance - threshold for acceptible distance
%                  obj.n1_oversampled (length of oversampled projection 1st dim)
%                  obj.n2_oversampled (length of oversampled projection 2nd dim)
%                  obj.DFT_CentroSymmetricity - 0 for complex-valued reconstruction
%                                        1 for real-valued reconstruction,
%                                        centrosymmetry will be enforced to
%                                        save calculation time
%                  obj.DFT_doGPU - 0 for not using GPU
%                           1 for using GPU
%                  obj.doCTFcorrection - 0 for not doing CTF correction
%                                        1 for doing CTF correction
%                  obj.CTFparameters - CTF parameters if doing CTF correction
%
%
%
% Second version date: 2015. 7. 7. (YY)
% Change: 1. Now this code properly process arbitrary-sized input projections
%         and oversampling. nx_ori, ny_ori, nx, ny can be arbitrary positive
%         integer, and it does not matter if they are even or odd.
%         2. This code assumes that the pixels of original projections are
%         "square", i.e. the pixel resolution is the same in both
%         directions. This version of code does not work if pixels are rectangular.
%         3. Implemented spacial frequency dependent interpolation
%         threshold (turn on and off using ThreshType argument)
%
% Thrid version date: 2015. 7. 9. (YY)
% Change: 1. Modified dimensional convention (1st dim: x, 2nd dim: y,
%                                            3rd dim: z)
%         2. Rotation matrix is now Z(phi)*X(theta)*Z(psi), instead of
%         previous version [Z(phi)*Y(theta)*Z(psi)].
%
%         3. Zero degree projection is in XY plane, not XZ plane as before.
%
% Fourth version date: 2016. 4. 11. (YY)
% Change: 1. Cleaned up the code a little bit
%
%         2. Made switch for centrosymmetricity, in case of complex
%         reconstruction
%
%         3. CTF correction
%
%         4. inline C function for speedup
%
% Sixth version date: 2016. 6. 26. (YY)
% Change: 1. Cleaned up the code a little bit
%
%         2. Made siwtches for CTF correction
%
%         3. Wiener filter CTF correction
%
% Seventh version date: 2016. 8. 11. (YY)
% Change: 1. the output should be nx x ny x nx array (y is the rotation
%              axis)
%         2. fixed bug for correctly determining the cutoff sphere
%
% Eighth version date: 2016. 8. 24. (YY)
% Change: 1. C function disabled because it is actually slower
%         2. Implemented GPU capability
%
% Nineth version date: 2016. 8. 25. (YY)
% Change: 1. Made the function consistent with AJ's convention
%
% Class version date: 2016. 12. 18. (YY)
% Change: 1. Made the fuction for use in GENFIRE_Class
%         2. Separated CTF correction and centrosymmetry part as separate
%         functions
%         3. Removed confidence weight calculation and SD calculation

function obj = fillInFourierGrid_DFT6(obj)

tic

% original projection dimensions
n1_ori = obj.Dim1;
n2_ori = obj.Dim2;
n1 = obj.n1_oversampled;
n2 = obj.n2_oversampled;

% if distance below minInvThresh, minInvThresh will be used
% this is to prevent division by zero
minInvThresh = 0.0000001;
cutoff_dist = obj.interpolationCutoffDistance;

% initialize normal vectors and rotation matrices array
normVECs = zeros(size(obj.InputProjections,3),3,'single');
rotMATs = zeros(3,3,size(obj.InputProjections,3),'single');

phis = obj.InputAngles(:,1);
thetas = obj.InputAngles(:,2);
psis = obj.InputAngles(:,3);

% calculate rotation matrices and normal vectors from the rotation matrices
for i=1:size(obj.InputProjections,3)
    
    rotmat1 = MatrixQuaternionRot(obj.vector1,phis(i));
    rotmat2 = MatrixQuaternionRot(obj.vector2,thetas(i));
    rotmat3 = MatrixQuaternionRot(obj.vector3,psis(i));
    
    rotMATs(:,:,i) =  (rotmat1*rotmat2*rotmat3);
    
    init_normvec = [0 0 1];
    normVECs(i,:) = squeeze(rotMATs(:,:,i))*init_normvec';
end

% initiate Fourier space indices
k1 = single((-1*ceil((n1-1)/2):1:floor((n1-1)/2)) );
k2 = single((-1*ceil((n2-1)/2):1:floor((n2-1)/2)) );
k3 = k1;
nc1 = floor(n1/2)+1;
nc2 = floor(n2/2)+1;

% Fourier grid
% in case of centrosymmetry, only half of k1 will be interpolated
% and centrosymmetricity will be enforced later
[K2, K1, K3] = meshgrid(k2,k1,k3);
K1 = K1(:)';
K2 = K2(:)';
K3 = K3(:)';

% initialize variables
%FS = zeros(size(K1),'single'); % Fourier points

%Numpt = zeros(size(K1),'single'); % array to store how many points found per Fourier point
%invSumTotWeight = zeros(size(K1),'single'); % array to store sum of weights of Fourier point

% initiate Fourier space indices
k1_ori = single(-1*ceil((n1_ori-1)/2):1:floor((n1_ori-1)/2)) ;
k2_ori = single(-1*ceil((n2_ori-1)/2):1:floor((n2_ori-1)/2)) ;

[K20, K10] = meshgrid(k2_ori,k1_ori);
[dim1,dim2] = size(K10);

if obj.DFT_doGPU
    K10G = gpuArray(K10(:));
    K20G = gpuArray(K20(:));
end

master_ind = [];
master_val = [];
master_dist= [];

%% DFT method
%
measuredX1 = [];
measuredY1 = [];
measuredZ1 = [];
fprintf('Computing DFT\n');
for p=1:size(obj.InputProjections,3)
    fprintf('Projection %d\n',p);
    % current projection
    curr_proj = squeeze(obj.InputProjections(:,:,p));
    %normVECs_p = normVECs(p,:); normVECs_p = normVECs_p/norm(normVECs_p);
    
    %[K2, K1, K3] = meshgrid(k2,k1,k3);
    % obtain points-to-plane distance
    D = distancePointsPlane_YY([K1; K2; K3], normVECs(p,:));
    % D = abs(normVECs_p*[K1;K2;K3]); % the same result
    
    % find Fourier points within the threshold
    Dind = find(D < obj.interpolationCutoffDistance);
    
    % Find the closest points to this current plane, CP: size 3xn
    CP = closestpoint(normVECs(p,:)',0,[K1(Dind); K2(Dind); K3(Dind)]);
    
    % Rotate current plane to zero degree
    CP_plane = (squeeze(rotMATs(:,:,p)))\CP;
    
    % picked closest point which is within the projection plain, x coordinate must be zero after rotation
    if sum(abs(CP_plane(3,:)) > 0.0001) > 0
        fprintf(1,'something wrong!\n');
    end
    
    %{
    tic
    CP_plane1 = (squeeze(rotMATs(:,:,p)))\[K1(:) K2(:) K3(:)]';
    Dind1 = (abs(CP_plane1(3,:)) <  obj.interpolationCutoffDistance);
    CP_plane1 = CP_plane1(:,Dind1);
    toc
    %[size(CP_plane), size(CP_plane1)]
    sum(sum(abs(CP_plane1(1:2,:) - CP_plane(1:2,:))))
    %}
    
    % consider Fourier points only within the resolution circle
    good_index = abs(CP_plane(1,:)) <= n1/2 & abs(CP_plane(2,:)) <= n2/2;
    len_good_index = nnz(good_index);
    %Gind = Dind(good_index);  % good indices
    G_CP_plane = CP_plane(:,abs(CP_plane(1,:)) <= n1/2 & abs(CP_plane(2,:)) <= n2/2 );  % good in-plane coordinates
    %[size(Gind), size(Dind), size(CP), size(good_index)]
    measuredX1 = [measuredX1, CP(1,good_index)];
    measuredY1 = [measuredY1, CP(2,good_index)];
    measuredZ1 = [measuredZ1, CP(3,good_index)];
    
    %determine the available memory in MB
    if obj.DFT_doGPU
        GPUinfo = gpuDevice();
        av_memory_size = round(GPUinfo.AvailableMemory/1000000);
    else
        
        % determine the available memory in MB
        if ispc % in case of Windows machine
            [~, SYSTEMVIEW] = memory;
            av_memory_size = SYSTEMVIEW.PhysicalMemory.Available / 1000000;
        elseif isunix && ~ismac  % in case of linux (or unix)
            [~,out]=system('cat /proc/meminfo | grep MemFree');
            av_memory_size=sscanf(out,'MemFree:          %d kB');
            av_memory_size = av_memory_size / 1000;
        else % in case of mac (I don't have mac now, to be implemented later)
            av_memory_size = 1000;
        end
    end
    
    memory_per_index = 40*length(curr_proj(:))/1000000;
    
    % determine block size for vectorized calculation
    block_size = max(1,floor(av_memory_size/memory_per_index));
    %block_size = 500;
    %cutnum = floor(length(Gind)/block_size);
    %cutrem = mod(length(Gind),block_size);
    cutloopnum = ceil(len_good_index/block_size);
    
    % loop over Fourier points within the threshold
    for i=1:cutloopnum
        curr_indices = ((i-1)*block_size+1):min(i*block_size,len_good_index);
        
        CTFcorr = ones(length(curr_indices),1);
        
        if obj.DFT_doGPU
            %
            G_CP_plane1_GPU_n = gpuArray(G_CP_plane(1,curr_indices)/n1);
            G_CP_plane2_GPU_n = gpuArray(G_CP_plane(2,curr_indices)/n2);
            curr_proj_GPU = gpuArray(curr_proj(:));
            
            % DFT calculation
            FpointsG = sum(bsxfun(@times, curr_proj_GPU, exp(-1*1i*2*pi*(K10G*G_CP_plane1_GPU_n+K20G*G_CP_plane2_GPU_n))),1);
            
            Fpoints = gather(FpointsG);
            Fpoints = CTFcorr.*Fpoints.';
            
            clear G_CP_plane1_GPU_n G_CP_plane2_GPU_n curr_proj_GPU FpointsG curr_proj_gpu xj yj
        else
            %Fpoints = CTFcorr.*sum(bsxfun(@times, curr_proj(:), exp(-1*1i*2*pi*(K10(:)*G_CP_plane(1,curr_indices)/n1+K20(:)*G_CP_plane(2,curr_indices)/n2))),1);
            %[size(K10),size(K20),size(curr_proj)]
            
            nj = length(curr_indices);
            xj = double(G_CP_plane(1,curr_indices)/n1*2*pi);
            yj = double(G_CP_plane(2,curr_indices)/n2*2*pi);
            Fpoints = nufft2d2(nj,xj(:),yj(:),-1,1e-6,dim1,dim2,double(curr_proj));
            
        end
        
        % collect indices and distance
        %CIND = Gind(curr_indices);        
        %master_ind = [master_ind,CIND];
        %master_dist = [master_dist,D(CIND)];
        master_val = [master_val;Fpoints];
        
        clear Fpoints
    end
end
%}
%% use FFT
%{
%l1=l2; k1=k2;
if size(master_ind,1)==1, master_ind = master_ind';end
if size(master_dist,1)==1, master_dist = master_dist';end
%[size(master_ind), size(master_val), size(master_dist) ]

NumProj = size(obj.InputProjections,3);
kMeasured = zeros(l1,l2,NumProj);
measuredX = zeros(l1*l2,NumProj,'single');
measuredY = zeros(l1*l2,NumProj,'single');
measuredZ = zeros(l1*l2,NumProj,'single');

[ky, kx] = meshgrid(k2,k1);
kz = zeros(1,l2*l1,'single');

%[size(kx),size(ky)]

ky = single(ky(:)');
kx = single(kx(:)');
%[min(kx),max(kx)]
%[min(ky),max(ky)]

for p = 1:NumProj;
    curr_proj = squeeze(obj.InputProjections(:,:,p));
    kMeasured_i = my_fft( My_paddzero(curr_proj,[l2 l2]));
    %size(kMeasured_i)
    kMeasured(:,:,p) = kMeasured_i(1:l1,:);
    
    Rt = squeeze(rotMATs(:,:,p));
    rotkCoords = Rt*[kx; ky; kz];%rotate coordinates
    
    measuredX(:,p) = rotkCoords(1,:);%rotated X
    measuredY(:,p) = rotkCoords(2,:);%rotated Y
    measuredZ(:,p) = rotkCoords(3,:);%rotated Z
end

if obj.allowMultipleGridMatches
    shiftMax = round(cutoff_dist);
else shiftMax = 0;
end
%}
%{
measuredX = [measuredX(:); measuredX1(:)];
measuredY = [measuredY(:); measuredY1(:)];
measuredZ = [measuredZ(:); measuredZ1(:)];
kMeasured = [kMeasured(:); master_val];
%}
measuredX =  measuredX1(:);
measuredY =  measuredY1(:);
measuredZ =  measuredZ1(:);
kMeasured =  master_val;
%% Gaussian kernel 
fprintf('Interpolation using RBF\n');
sigma = obj.sigma_RBF;
s_size = 24;

% define number of maximum points in a sub-domain. It is also the largest
% matrix size that our computer can solve efficiently
max_points = 18000;
num_points = max_points;
while num_points>=max_points
    index_ijk = find( abs(measuredX)<=s_size/2 & abs(measuredZ)<=s_size/2 & abs(measuredY)<=s_size/2 );
    num_points = length(index_ijk);
    if num_points>=max_points
        s_size = s_size-2;
    end
end
fprintf('partition size = %d\n',s_size);

rangeX = s_size/2:s_size:(n1-nc1); remainder = n1-nc1-rangeX(end);
rangeX = [-flip(rangeX),rangeX]; 
if remainder > s_size/2
    rangeX = [-nc1+1,rangeX,n1-nc1];
else
    rangeX(1) = -nc1+1; rangeX(end)=n1-nc1;
end
lenX = length(rangeX);

rangeY = s_size/2:s_size:(n2-nc2); remainder = n2-nc2-rangeY(end);
rangeY = [-flip(rangeY),rangeY];
if remainder > s_size/2
    rangeY = [-nc2+1,rangeY,n2-nc2];
else
    rangeY(1) = -nc2+1; rangeY(end)=n2-nc2;
end
lenY = length(rangeY);

[measuredXYZ,~,index_XYZ] = unique([measuredX,measuredY,measuredZ],'rows');
measuredX = measuredXYZ(:,1);
measuredY = measuredXYZ(:,2);
measuredZ = measuredXYZ(:,3);
clear measuredXYZ
%kMeasured = kMeasured(index_XYZ_unique);
kMeasured = accumarray(index_XYZ, kMeasured, [] ,@mean);
masterInd=[];
masterVals = [];
%{
if obj.DFT_doGPU
    measuredX = gpuArray(measuredX);
    measuredY = gpuArray(measuredY);
    measuredZ = gpuArray(measuredZ);
    kMeasured = gpuArray(kMeasured);
end
%}
% parloop(16); %uncomment this line and use parfor instead on hoffman2
parfor t=1:(lenX-1)^2*(lenY-1)
    if mod(t,(lenX-1)^2)==1, fprintf('#### %d ####\n',ceil(t/(lenX-1)^2));end    
    
    % index t to subcript (i,j,k)
    [i,j,k] = ind2sub([lenX-1, lenX-1, lenY-1],t);
    
    index_ijk = find( measuredX>=rangeX(i) & measuredX<=rangeX(i+1) & measuredZ>=rangeX(j) & measuredZ<=rangeX(j+1) & measuredY>=rangeY(k) & measuredY<=rangeY(k+1) );
   
    if k==1 && mod(t,lenX-1)==1,fprintf('%d.length = ',j);end
    if ~isempty(index_ijk)
        if k==1, fprintf('%d, ',length(index_ijk)); end
        
        measuredX_i = measuredX(index_ijk);
        measuredY_i = measuredY(index_ijk);
        measuredZ_i = measuredZ(index_ijk);
        measuredk_i = kMeasured(index_ijk);
        
        rx = bsxfun(@minus, measuredX_i,measuredX_i');
        ry = bsxfun(@minus, measuredY_i,measuredY_i');
        rz = bsxfun(@minus, measuredZ_i,measuredZ_i');
        % you can use different RBF
        A = exp(-sigma^2*sqrt(rx.^2 + ry.^2 + rz.^2));
        
        phi = A\measuredk_i;
        %A(A<exp(-2*sigma^2))=0;
        %[phi,~] = pcg(A,measuredk_i,1e-12);
        
        tmpX_i = round(measuredX_i);
        tmpY_i = round(measuredY_i);
        tmpZ_i = round(measuredZ_i);
        dist = sqrt(abs(measuredX_i-tmpX_i).^2+abs(measuredY_i-tmpY_i).^2+abs(measuredZ_i-tmpZ_i).^2);
        goodInd = ~(tmpX_i>n1-nc1| tmpX_i<1-nc1| tmpY_i>n2-nc2| tmpY_i<1-nc2| tmpZ_i>n1-nc1| tmpZ_i<1-nc1) & dist<cutoff_dist;
        tmpX_i = tmpX_i(goodInd);
        tmpY_i = tmpY_i(goodInd);
        tmpZ_i = tmpZ_i(goodInd);
        masterInd_i = sub2ind([n1 n2 n1],tmpX_i+nc1,tmpY_i+nc2,tmpZ_i+nc1);
        masterInd = [masterInd;masterInd_i];
        
        rx = bsxfun(@minus, tmpX_i, measuredX_i');
        ry = bsxfun(@minus, tmpY_i, measuredY_i');
        rz = bsxfun(@minus, tmpZ_i, measuredZ_i');
        A = exp(-sigma^2*sqrt(rx.^2 + ry.^2 + rz.^2));
        masterVal_i = A*phi;
        masterVals = [masterVals;masterVal_i];
    end
    if k==1 && mod(t,lenX-1)==0,fprintf('\n');end
    
end
% delete(gcp('nocreate'));
masterVals(isnan(masterVals))=0;
obj.measuredK = accumarray(masterInd,masterVals,[n1*n2*n1 1],@mean);
%}
%% reshape and make Hermitian matrix
obj.measuredK = reshape(obj.measuredK,[n1 n2 n1]);
obj.measuredK = hermitianSymmetrize(obj.measuredK);

obj.recIFFT = My_stripzero(real(my_ifft(obj.measuredK)), [obj.Dim1 obj.Dim2 obj.Dim1]);

timeTakenToFillInGrid = toc;
timeTakenToFillInGrid = round(10*timeTakenToFillInGrid)./10;
fprintf('GENFIRE: Fourier grid assembled in %.12g seconds.\n\n',timeTakenToFillInGrid);

end



function D = distancePointsPlane_YY(points, normvec)
%distancePointsPlane_YY unsigned distances betwen 3D points and a plane
% through origin
%
%   D = distancePointsPlane_YY(point, normvec)
%   Returns the euclidean distance between points and a plane going through origin with normal vector normvec,
%   given by:
%   points : (3 x n) array of 3D vectors
%   normvec : (3 x 1) or (1 x 3) array of normal vector
%   D     : (1 x n) vector

%
%   ---------
%   author : Y. Yang, UCLA Physics and Astronomy
%   created the 05/03/2015.
%

% normalized plane normal
normvec = normvec(:) / norm(normvec);
D = abs(points(1,:)*normvec(1) + points(2,:)*normvec(2) + points(3,:)*normvec(3));

end


function x = closestpoint(n, d, p)
% n is the vector [A,B,C] that defines the plane
% d is the distance of the plane from the origin
% p is the point  [P,Q,R]
if size(p,2) == 1
    v = (d - sum(p.*n)) / sum(n.*n);
    x = p + v * n;
else
    nr = repmat(n,[1 size(p,2)]);
    v = (d - sum(p.*nr,1)) / sum(n.*n);
    x = p + repmat(v,[3 1]) .* nr;
end
end


function dd = MatrixQuaternionRot(vector,theta)

theta = theta*pi/180;
vector = vector/sqrt(dot(vector,vector));
w = cos(theta/2); x = -sin(theta/2)*vector(1); y = -sin(theta/2)*vector(2); z = -sin(theta/2)*vector(3);
RotM = [1-2*y^2-2*z^2 2*x*y+2*w*z 2*x*z-2*w*y;
    2*x*y-2*w*z 1-2*x^2-2*z^2 2*y*z+2*w*x;
    2*x*z+2*w*y 2*y*z-2*w*x 1-2*x^2-2*y^2;];

dd = RotM;
end

