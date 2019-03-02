%%  GENFIRE_iterate dr %%

function obj = reconstruct_dr(obj)

% paddedSupport = My_paddzero(obj.Support, [obj.n1_oversampled obj.n2_oversampled obj.n1_oversampled]); 
paddedSupport = My_paddzero(obj.Support, size(obj.measuredK)); 

Q = make_Kspace_indices(paddedSupport);
resRange = -0.05;%thickness of resolution ring to use for removal of datapoints for Rfree test

constraintIndicators = zeros(size(Q)); %% first reconstruction uses resolution extension/suppression, where lower resolution information is enforced
%%initially and the maximum enforced resolution increases. This is followed by resolution where suppression, which is the same process run backwards, so at the final iteration 
%%only the lowest resolution is being enforced again.
constraintIndicators(obj.measuredK~=0 & obj.measuredK_mask) = 1-Q(obj.measuredK~=0 & obj.measuredK_mask);%make lower resolution have higher confidence. Higher confidence means enforced earlier in the reconstruction and for longer overall than low confidence

%remove datapoints for Rfree calculation

if obj.calculate_Rfree==1
  spatialFrequencyForRfree = linspace(0,1,obj.numBinsRfree+1);%compute spatial frequency bins
  for shellNum = 1:obj.numBinsRfree %loop over each frequency shell

      measuredPointInd_complex = find(obj.measuredK~=0&Q>=(spatialFrequencyForRfree(shellNum)+resRange)&Q<spatialFrequencyForRfree(shellNum+1)); %candidate values for Rfree_complex
      
      if ~isempty(obj.RandomSeed)
          rng(obj.RandomSeed);
      end
      P = randperm(numel(measuredPointInd_complex)); %shuffle values
      
      measuredPointInd_complex = measuredPointInd_complex(P); %apply shuffle
      cutoffInd_complex = floor(numel(measuredPointInd_complex).*obj.percentValuesForRfree); %take indices for 5% of measured data
      if cutoffInd_complex == 0 %make sure to include at least one point
          cutoffInd_complex = 1;
      end
      R_freeInd_complex{shellNum} = measuredPointInd_complex(1:cutoffInd_complex);%take complex value for 5% of measured data
      R_freeInd_shifted_complex{shellNum} = My_iffshift3_ind(size(paddedSupport),R_freeInd_complex{shellNum});       
      %now create a temporary set of constraints that have this 5% of
      %datapoints removed
      obj.measuredK_mask(R_freeInd_complex{shellNum}) = false;
      R_freeVals_complex{shellNum} = obj.measuredK(R_freeInd_complex{shellNum});
  end
end

%run the actual reconstruction
fprintf('GENFIRE: Reconstructing... \n\n');

tic;

if isempty(obj.initialObject)
    initialObject = zeros(size(paddedSupport));
else
    initialObject = My_paddzero(obj.InitialObject, [obj.n1_oversampled obj.n2_oversampled obj.n1_oversampled]);
end

numIterations = obj.numIterations;
enforce_positivity = obj.constraintPositivity;
enforce_support = obj.constraintSupport;

% numIterations,initialObject,support,measuredK,constraintIndicators,constraintEnforcementDelayIndicators,
% 
% R_freeInd_complex,R_freeVals_complex, 

bestErr = 1e30;%initialize best error

if obj.calculate_Rfree==1
    obj.Rfree_complex = -1*ones(1,numIterations,'single');%% initialize Rfree_complex curve , -1 is a flag that means undefined
end
obj.errK = zeros(1,numIterations,'single');

%prefetch indices to use for error metric to avoid having to lookup each
%iteration
errInd = find(obj.measuredK~=0&obj.measuredK_mask);
errInd_shifted = My_iffshift3_ind(size(paddedSupport),errInd);
        
%determine how to spread the provided weighting cutoffs over the iterations
iterationNumsToChangeCutoff = round(linspace(1,numIterations,numel(obj.constraintEnforcementDelayIndicators)));

currentCutoffNum = 1;

% do ifftshift and fftshift only at before and after iteration
% and avoid using ifftshift and fftshift during the iteration to save time
paddedSupport = ifftshift(paddedSupport);
initialObject = ifftshift(initialObject);

u = initialObject;
for iterationNum = 1:numIterations
    if iterationNum == iterationNumsToChangeCutoff(currentCutoffNum)
        currentCutoffNum = find(iterationNumsToChangeCutoff==iterationNum,1,'last');
        constraintInd_complex = find(constraintIndicators>(obj.constraintEnforcementDelayIndicators(currentCutoffNum))&obj.measuredK~=0&obj.measuredK_mask);
        constraintInd_complex_shifted = My_iffshift3_ind(size(paddedSupport),constraintInd_complex);        
        currentCutoffNum = currentCutoffNum+1;
        bestErr = 1e30;%reset best error
    end
    
    switch obj.ds_type
        case 1
            ds = obj.ds_value;
        case 2
            ds = 1 - 1*(1-sqrt((numIterations-iterationNum)/numIterations));
    end
    switch obj.dt_type
        case 1
            dt = obj.dt_value;
        case 2
            dt = 0.1 + 0.3*(1-sqrt((numIterations-iterationNum)/numIterations));
    end
    
    if enforce_positivity
        initialObject(initialObject<0) = 0; %enforce positivity
    end
    if enforce_support
        initialObject = initialObject.*paddedSupport;%enforce support
    end

    k = fftn(initialObject);%take FFT of initial object
    %monitor error
    obj.errK(iterationNum) = sum(abs(abs(k(errInd_shifted))-abs(obj.measuredK(errInd))))./sum(abs(obj.measuredK(errInd)));

    if obj.calculate_Rfree==1 %if values have been withheld from measuredK for monitoring R_free, check them accordingly
        if ~isempty(R_freeInd_complex)
            %calculate Rfree in each resolution shell
            for shellNum = 1:numel(R_freeInd_complex)
                %tmpInd =R_freeInd_complex{shellNum};
                tmpInd_shifted =R_freeInd_shifted_complex{shellNum};
                tmpVals = R_freeVals_complex{shellNum};                
                obj.Rfree_complex(shellNum,iterationNum) = sum(abs(k(tmpInd_shifted)-tmpVals))./sum(abs(tmpVals));
            end
        end
    end
    
    if obj.errK(iterationNum)<bestErr %if current reconstruction has better error, update best error and best reconstruction
    %     fprintf('GENFIRE: new best object, iteration %d\n',iterationNum)
        bestErr = obj.errK(iterationNum);
        obj.reconstruction = initialObject;
    end

    if obj.save_temp == 1 && mod(iterationNum,obj.save_loopLength) == 0
        temp_reconstruction = My_stripzero(fftshift(obj.reconstruction),[obj.Dim1 obj.Dim2 obj.Dim1]);
        temp_filename = [obj.saveFilename,num2str(iterationNum),'.mat'];
        save(temp_filename, 'temp_reconstruction');
    end
    
    fprintf('GENFIRE: Iteration %d: Error = %d\n',iterationNum, obj.errK(iterationNum));
    %enforce Fourier constraint
%     k(constraintInd_complex_shifted) = obj.measuredK(constraintInd_complex);
    k(constraintInd_complex_shifted) = dt*k(constraintInd_complex_shifted) + (1-dt)*obj.measuredK(constraintInd_complex);
    u_K = ifftn(k);
%     u_K = real(ifftn(k));
    initialObject = (1+ds)*u_K - ds*u;
    
%     initialObject = real(ifftn(k));%obtain next object with IFFT
    initialObject = real(initialObject);%obtain next object with IFFT
    u = initialObject + ds*(u - u_K);
end

reconstructionTime = toc;
reconstructionTime = round(10*reconstructionTime)./10;
fprintf('GENFIRE: Reconstruction completed in %.12g seconds.\n\n',reconstructionTime);

obj.reconstruction = My_stripzero(fftshift(obj.reconstruction),[obj.Dim1 obj.Dim2 obj.Dim1]);

end
