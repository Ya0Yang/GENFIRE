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

function obj = fillInFourierGrid_DFT4(obj)

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

%obj.DFT_CentroSymmetricity=0;
% initiate Fourier space indices
if obj.DFT_CentroSymmetricity
    k1 = single((-1*floor(n1/2):1:0)) ;
    n_k1 = single(floor(n1/2));
    
    k2 = single( -1*floor(n2/2):1:floor(n2/2) ) ;
    k3 = single( -1*floor(n1/2):1:floor(n1/2) ) ;    
else
    k1 = single((-1*ceil((n1-1)/2):1:floor((n1-1)/2)) );
    k2 = single((-1*ceil((n2-1)/2):1:floor((n2-1)/2)) );
    k3 = k1;
end
nc = floor(n1/2)+1;
%[length(k1),length(k2),length(k3)]
l1 = length(k1);
l2 = length(k2);
l3 = length(k3);

% Fourier grid
% in case of centrosymmetry, only half of k1 will be interpolated
% and centrosymmetricity will be enforced later
[K2, K1, K3] = meshgrid(k2,k1,k3);
K1 = K1(:)';
K2 = K2(:)';
K3 = K3(:)';
%[max(K2(:)),min(K2(:))] %= [-96,96]
%[min(K1(:)),max(K1(:))]

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
for p=1:size(obj.InputProjections,3)
    % current projection
    curr_proj = squeeze(obj.InputProjections(:,:,p));
    %normVECs_p = normVECs(p,:); normVECs_p = normVECs_p/norm(normVECs_p);
    
    %[K2, K1, K3] = meshgrid(k2,k1,k3);
    % obtain points-to-plane distance    
    D = distancePointsPlane_YY([K1; K2; K3], normVECs(p,:));
    % D = abs(normVECs_p*[K1;K2;K3]); % the same result   
    
    % find Fourier points within the threshold
    Dind = find(D < obj.interpolationCutoffDistance);

    % rotate the plane to zero degree
    % size 3xn
    CP = closestpoint(normVECs(p,:)',0,[K1(Dind); K2(Dind); K3(Dind)]); 

    
    %toc
    CP_plane = (squeeze(rotMATs(:,:,p)))\CP;

    %clear KX KY KZ
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
    %cutnum = floor(length(Gind)/block_size);
    %cutrem = mod(length(Gind),block_size);
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
            %
            G_CP_plane1_GPU_n = gpuArray(G_CP_plane(1,curr_indices)/n1);
            G_CP_plane2_GPU_n = gpuArray(G_CP_plane(2,curr_indices)/n2);
            curr_proj_GPU = gpuArray(curr_proj(:));            
            
            % DFT calculation
            FpointsG = sum(bsxfun(@times, curr_proj_GPU, exp(-1*1i*2*pi*(K10G*G_CP_plane1_GPU_n+K20G*G_CP_plane2_GPU_n))),1);
            
            Fpoints = gather(FpointsG);
            Fpoints = CTFcorr.*Fpoints.';
            
            
            %
            %nj = length(curr_indices);
            %xj = gpuArray(double(G_CP_plane(1,curr_indices)/n1*2*pi));
            %yj = gpuArray(double(G_CP_plane(2,curr_indices)/n2*2*pi));
            %curr_proj_gpu = gpuArray(double(curr_proj));
            %Fpoints = nufft2d2(nj,xj(:),yj(:),-1,1e-6,dim1,dim2,curr_proj_gpu);
            %Fpoints = gather(Fpoints);
            %Fpoints = CTFcorr.*Fpoints;            
            %
             clear G_CP_plane1_GPU_n G_CP_plane2_GPU_n curr_proj_GPU FpointsG curr_proj_gpu xj yj 
        else
            %Fpoints = CTFcorr.*sum(bsxfun(@times, curr_proj(:), exp(-1*1i*2*pi*(K10(:)*G_CP_plane(1,curr_indices)/n1+K20(:)*G_CP_plane(2,curr_indices)/n2))),1);            
            %[size(K10),size(K20),size(curr_proj)]

            nj = length(curr_indices);
            xj = double(G_CP_plane(1,curr_indices)/n1*2*pi);
            yj = double(G_CP_plane(2,curr_indices)/n2*2*pi);
            %[min(xj),max(xj)]
            %[min(yj),max(yj)]
            Fpoints = nufft2d2(nj,xj(:),yj(:),-1,1e-6,dim1,dim2,double(curr_proj));
            %Fpoints=Fpoints';
            %sum(abs(Fpoints(:)-Fpoints2(:)))/sum(abs(Fpoints(:)))
            
        end        
        
        %weighted avearaging
        CIND = Gind(curr_indices);                 
        %disp(length(CIND))
        
        master_ind = [master_ind,CIND];
        master_dist = [master_dist,D(CIND)];
        master_val = [master_val;Fpoints];        
                
        %{
        currDist = D(CIND);
        currDist(currDist < minInvThresh) = minInvThresh; % if distance smaller than minInvThresh, put minInvThresh (to prevent divison by zero)
        currInvDist = 1./ currDist;           % inverse distance
        
        % re-average inverse distance
        FS(CIND) = FS(CIND).* invSumTotWeight(CIND) + currInvDist.*Fpoints;
        invSumTotWeight(CIND) = invSumTotWeight(CIND) + currInvDist;
        FS(CIND) = FS(CIND) ./ invSumTotWeight(CIND);
        %}
        clear Fpoints
    end   
end
master_val1 = master_val;
%}
%% use FFT
%
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

interpolationCutoffDistance = obj.interpolationCutoffDistance;
if obj.allowMultipleGridMatches
    shiftMax = round(interpolationCutoffDistance);
else shiftMax = 0;
end

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
            %[min(tmpX(:)),max(tmpX(:))]
            %[min(tmpY(:)),max(tmpY(:))]
            %[min(tmpZ(:)),max(tmpZ(:))]
            goodInd = (~(tmpX>l1|tmpX<1|tmpY>l2|tmpY<1|tmpZ>l2|tmpZ<1)) & distances<=interpolationCutoffDistance;%find candidate values            
            master_ind = [master_ind; sub2ind([l1 l2 l2],tmpX(goodInd),tmpY(goodInd),tmpZ(goodInd))]; %append values to lists
            master_val = [master_val; tmpVals(goodInd)];
            master_dist = [master_dist; distances(goodInd)];            
        end
    end
end
%[size(master_ind), size(master_val), size(master_dist) ]
%}
%% take average
sigma = obj.sigma_GaussKernel;
if size(master_ind,1 )==1, master_ind  = master_ind'; end
if size(master_dist,1)==1, master_dist = master_dist'; end
%master_val  = transpose(master_val);

index_exact = master_dist < minInvThresh;
%master_dist(index_exact)=minInvThresh;
%master_dist = 1./master_dist;
master_dist(index_exact)=0;
%master_dist(~index_exact) = 1./master_dist(~index_exact);
% kernel smoothing
master_dist = exp(-sigma*master_dist);
%master_dist(~index_exact) = exp(-100*master_dist(~index_exact));


%k_exact = master_val(index_exact);
%index_k = master_ind(index_exact);


FS2        = accumarray(master_ind, master_val.*master_dist, [l1*l2*l3 1]);
sum_weight = accumarray(master_ind, master_dist, [l1*l2*l3 1]);
index_interp = sum_weight>0;
%[nnz(index_interp), nnz(FS2)]


%numer = FS.*invSumTotWeight;
%sum(abs(invSumTotWeight(:)-sum_weight(:))) /sum(abs(invSumTotWeight(:)))
%sum(abs(numer(:)-FS2(:))) /sum(abs(numer(:)))
FS2(index_interp) = FS2(index_interp) ./ sum_weight(index_interp);
%FS2(~index_interp)=0;
%FS2(index_k) = k_exact;

FS2 = reshape(FS2,l1,l2,l3);
%[size(FS),size(FS2)]

%% triangulation
%{
[YY,XX,ZZ] = meshgrid(double(k2),double(k1),double(k2));
index_interp = sum_weight>0;
XX=XX(index_interp);YY=YY(index_interp);ZZ=ZZ(index_interp);
%measuredK=griddata(double(measuredX),double(measuredY),double(measuredZ),double(kMeasured),XX,YY,ZZ);
%measuredK=griddata(double(measuredX1),double(measuredY1),double(measuredZ1),double(master_val),XX,YY,ZZ,'nearest');
measuredK=griddata(double([measuredX(:);measuredX1']),double([measuredY(:);measuredY1']),double([measuredZ(:);measuredZ1']),double([kMeasured(:);master_val1]),XX,YY,ZZ);
%[size(measuredX),size(measuredX1),size(master_val)]
measuredK(isnan(measuredK))=0;
FS3 = zeros([l1,l2,l2],'single');
FS3(index_interp) = measuredK;
FS3(index_k) = k_exact;
FS3 = reshape(FS3,[l1,l2,l3]);
FS2 = FS3;
%}
%%
%fprintf('diff = %.7f\n',sum(abs(FS2(:)-FS(:)))/sum(abs(FS(:))));
%[n1,n2,size(FS2)]
% enforce centrosymmetricity
if obj.DFT_CentroSymmetricity
    obj = obj.CentroSymmetricity(FS2, n1, n_k1, n2);
    %obj.measuredK = obj.measuredK(1:end-1,1:end-1,1:end-1);
    %size(obj.measuredK)
else
    obj.measuredK = reshape(FS2,n1,n2,n1);    
    %obj.measuredK = reshape(FS2,n1+1,n2+1,n1+1); 
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

