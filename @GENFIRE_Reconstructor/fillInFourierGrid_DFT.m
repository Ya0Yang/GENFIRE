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
if obj.DFT_CentroSymmetricity
    k1 = single((-1*floor(n1/2):1:0)) ;
    n_k1 = single(floor(n1/2));
    
    k2 = single((-1*floor(n2/2):1:floor(n2/2))) ;
    k3 = single((-1*floor(n1/2):1:floor(n1/2))) ;
    
else
    k1 = single((-1*ceil((n1-1)/2):1:floor((n1-1)/2)) );
    k2 = single((-1*ceil((n2-1)/2):1:floor((n2-1)/2)) );
    k3 = k1;
end
l1=length(k1);
l2=length(k2);
l3=length(k3);

% Fourier grid
% in case of centrosymmetry, only half of k1 will be interpolated
% and centrosymmetricity will be enforced later
[K2, K1, K3] = meshgrid(k2,k1,k3);
K1 = K1(:)';
K2 = K2(:)';
K3 = K3(:)';

% initiate Fourier space indices
k1_ori = single(-1*ceil((n1_ori-1)/2):1:floor((n1_ori-1)/2)) ;
k2_ori = single(-1*ceil((n2_ori-1)/2):1:floor((n2_ori-1)/2)) ;

[K20, K10] = meshgrid(k2_ori,k1_ori);
[dim1,dim2]=size(K10);

if obj.DFT_doGPU
    K10G = gpuArray(K10(:));
    K20G = gpuArray(K20(:));
end

master_ind = [];
master_val = [];
master_dist= [];

%% DFT method
fprintf('Computing DFT:\n');
measuredX1 = [];
measuredY1 = [];
measuredZ1 = [];
for p=1:size(obj.InputProjections,3)
    % current projection
    fprintf('Projection %d\n',p);
    curr_proj = squeeze(obj.InputProjections(:,:,p));
    
    %[K2, K1, K3] = meshgrid(k2,k1,k3);
    
    % obtain points-to-plane distance
    D = distancePointsPlane_YY([K1; K2; K3], normVECs(p,:));
    
    % find Fourier points within the threshold
    Dind = find(D < obj.interpolationCutoffDistance);
    
    % rotate the plane to zero degree
    CP = closestpoint(normVECs(p,:)',0,[K1(Dind); K2(Dind); K3(Dind)]);
    CP_plane = (squeeze(rotMATs(:,:,p)))\CP;
    
    % picked closest point which is within the projection plain, x coordinate must be zero after rotation
    if sum(abs(CP_plane(3,:)) > 0.0001) > 0
        fprintf(1,'something wrong!\n');
    end
    
    % consider Fourier points only within the resolution circle
    good_index = abs(CP_plane(1,:)) <= n1/2 & abs(CP_plane(2,:)) <= n2/2;
    Gind = Dind(good_index);  % good indices
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
    cutloopnum = ceil(length(Gind)/block_size);
    
    
    % loop over Fourier points within the threshold
    for i=1:cutloopnum
        curr_indices = ((i-1)*block_size+1):min(i*block_size,length(Gind));
        
        % CTF correction
%         if obj.doCTFcorrection  % to be implemented as subfunction
%             CTFcorr = ones(length(curr_indices),1);
%         else
            CTFcorr = ones(length(curr_indices),1);
%         end
        
        if obj.DFT_doGPU
            G_CP_plane1_GPU_n = gpuArray(G_CP_plane(1,curr_indices)/n1);
            G_CP_plane2_GPU_n = gpuArray(G_CP_plane(2,curr_indices)/n2);
            curr_proj_GPU = gpuArray(curr_proj(:));
            
            
            % DFT calculation
            FpointsG = sum(bsxfun(@times, curr_proj_GPU, exp(-1*1i*2*pi*(K10G*G_CP_plane1_GPU_n+K20G*G_CP_plane2_GPU_n))),1);            
            
            Fpoints = gather(FpointsG);
            Fpoints = CTFcorr.*Fpoints.';
            
            clear G_CP_plane1_GPU_n G_CP_plane2_GPU_n curr_proj_GPU FpointsG
        else
            nj = length(curr_indices);
            xj = double(G_CP_plane(1,curr_indices)/n1*2*pi);
            yj = double(G_CP_plane(2,curr_indices)/n2*2*pi);
            Fpoints = nufft2d2(nj,xj(:),yj(:),-1,1e-6,dim1,dim2,double(curr_proj));
        end
        
        %weighted avearaging
        CIND = Gind(curr_indices);
        
        master_ind = [master_ind,CIND];
        master_dist = [master_dist,D(CIND)];
        master_val = [master_val;Fpoints];
        
        clear Fpoints
    end    
end
%% take average
fprintf('Computing average weight\n');
master_ind  = master_ind';
master_dist = master_dist';

index_exact = master_dist < minInvThresh;
%master_dist(index_exact)=minInvThresh;
%master_dist = 1./master_dist;
master_dist(index_exact)=0;
master_dist(~index_exact) = 1./master_dist(~index_exact);
% kernel smoothing
%master_dist = exp(-sigma^2*master_dist);
%master_dist(~index_exact) = exp(-100*master_dist(~index_exact));


k_exact = master_val(index_exact);
index_k = master_ind(index_exact);


FS        = accumarray(master_ind, master_val.*master_dist, [l1*l2*l3 1]);
sum_weight = accumarray(master_ind, master_dist, [l1*l2*l3 1]);
index_interp = sum_weight>0;

FS(index_interp) = FS(index_interp) ./ sum_weight(index_interp);
%FS2(~index_interp)=0;
FS(index_k) = k_exact;

FS = reshape(FS,l1,l2,l3);

%% enforce centrosymmetricity
if obj.DFT_CentroSymmetricity
    obj = obj.CentroSymmetricity(FS, n1, n_k1, n2);
else
    obj.measuredK = reshape(FS,n1,n2,n1);
end

clear FS

obj.recIFFT = My_stripzero(real(my_ifft(obj.measuredK)),[obj.Dim1 obj.Dim2 obj.Dim1]);

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

