classdef GENFIRE_Reconstructor
    
    properties
        % internal variables to be cleared after run
        InputProjections
        InputAngles
        measuredK
        measuredK_mask
        Support
        initialObject
        recIFFT
        
        
        % internal variables to be kept after run
        reconstruction
        final_rec
        errK
        Dim1  % projection size1
        Dim2  % projection size2
        n1_oversampled  % projection size1 after oversampling
        n2_oversampled  % projection size2 after oversampling
        NumProjs
        Rfree_complex
        constraintEnforcementDelayIndicators
        Rfactor
        Rarray
        simuProjs
        
        % filenames
        filename_Projections = ''
        filename_Angles = '';
        filename_Support = '';
        filename_InitialModel = '';
        filename_Results = './results/GENFIRE_rec.mat';
        
        % reconstruction parameters
        numIterations = 50;
        oversamplingRatio =3;
        interpolationCutoffDistance =.7;
        griddingMethod = 1; % 1 for FFT,  2 for DFT
        allowMultipleGridMatches = 1; % matters for FFT gridding only
        calculate_Rfree = 0; % close Rfree
        
        % axes vectors for phi, theta, psi
        vector1 = [0 0 1];
        vector2 = [0 1 0];
        vector3 = [1 0 0];
        
        % projections will be cropped to this size
        particleWindowSize

        % reconstruction constraints
        constraintEnforcementMode = 1; % Fourier enforcement weighting mode
        constraintPositivity = 1;
        constraintSupport = 1;
          
        % DFT related parameters
        DFT_CentroSymmetricity = 1;
        DFT_doGPU = 0;
        sigma_RBF = 0.5
        sigma_GaussKernel = 100
        
        % dt parameters for new dr method
        dt_type  = 1;
        dt_value = 0.1;
        ds_type = 1;
        ds_value = 0.5;
        Vol_ind =  zeros(3,2)
        smooth = 0;
        
        % save temp reconstruction
        save_temp = 0;
        save_loopLength = 50;
        saveFilename = './results/temp_rec';
    end
    
    methods
        
        % declare long methods in external files
        obj = fillInFourierGrid_DFT(obj)
        obj = fillInFourierGrid(obj)
        
        obj = reconstruct(obj)
        obj = reconstruct_dr(obj)
        obj = reconstruct_dr_relaxed(obj)
        obj = CentroSymmetricity(obj, FS, n1, n_k1, n2)
        
        % declare short methods in this file
        function obj = readFiles(obj)
            
            if FileExist(obj.filename_Projections)
                obj.InputProjections = importdata(obj.filename_Projections);
            else
                error('GENFIRE: Projections file does not exist!')
            end
            
            if FileExist(obj.filename_Angles)
                obj.InputAngles = importdata(obj.filename_Angles);
            else
                error('GENFIRE: Angles file does not exist!')
            end
            
            if ~isempty(obj.filename_Support)
                if FileExist(obj.filename_Support)
                    obj.Support = importdata(obj.filename_Support) ~=0 ;
                else
                    fprintf('GENFIRE: Support file does not exist!Making a support based on Projections.\n');
                    obj.Support = ones(size(obj.InputProjections,1),size(obj.InputProjections,2),size(obj.InputProjections,1)) ~=0 ;
                end
            end
            
            if ~isempty(obj.filename_InitialModel)
                if FileExist(obj.filename_InitialModel )
                    obj.initialObject = importdata(obj.filename_InitialModel);
                else
                    error('GENFIRE: Inital Model file does not exist!')
                end
            end
        end
        
        
        function obj = CheckPrepareData(obj)
            
            % set number of projections
            obj.NumProjs = size(obj.InputProjections,3);
            obj.Dim1 = size(obj.InputProjections,1);
            obj.Dim2 = size(obj.InputProjections,2);
            obj.particleWindowSize = [obj.Dim1, obj.Dim2];
            
            % input angle and projection size check
            if obj.NumProjs~=size(obj.InputAngles,1)
                error('GENFIRE: Number of projections and Angles does not match!')
            end
            
            % input angle check
            if size(obj.InputAngles,2) >3
                error('GENFIRE: Input Angle 2nd dimenstions larger than three!')
            end
            
            % if only one angle set is given, make them three Euler angles
            if size(obj.InputAngles,2) == 1
                obj.InputAngles = [zeros(obj.NumProjs ,1); obj.InputAngles; zeros(obj.NumProjs ,1)];
            end
            
            % check if initial object size matches with projection size
            if ~isempty(obj.initialObject)
                if sum(abs(size(obj.initialObject)-[obj.Dim1 obj.Dim2 obj.Dim1]))~=0
                    error('GENFIRE: Initial Model size does not match with projections!')
                end
            end
            
            % set support if support is not given
            if isempty(obj.Support)
                obj.Support = ones(obj.Dim1, obj.Dim2, obj.Dim1) ~= 0;
            end
            
            % check if gridding method is legitimate
            if obj.griddingMethod>6
                error('GENFIRE: Unrecognized gridding method.')
            end
            
            % set constraintEnforcementDelayIndicators depending on input mode
            switch obj.constraintEnforcementMode
                case 1
                    obj.constraintEnforcementDelayIndicators = [0.95:-0.1:-0.15 -10 -10 -10 -0.15:0.1:.95];
                case 2
                    obj.constraintEnforcementDelayIndicators = [0.95:-0.1:-0.15 -10 -10];
                case 3
                    obj.constraintEnforcementDelayIndicators = [-999, -999, -999, -999, -999];
                otherwise
                    error('GENFIRE: ERROR! constraintEnforcementMode value %d not understood',obj.constraintEnforcementMode)
            end
            
            % set oversampled array size
            obj.n1_oversampled = round(obj.Dim1 * obj.oversamplingRatio);
            obj.n2_oversampled = round(obj.Dim2 * obj.oversamplingRatio);
            
        end
        function obj = runGridding(obj)
            fprintf('GENFIRE: Assembling Fourier grid...\n\n');
            
            switch obj.griddingMethod
                case 0
                    obj = obj.fillInFourierGrid_DFT_old();
                case 1
                    obj = obj.fillInFourierGrid();
                case 2
                    obj = obj.fillInFourierGrid_DFT();
                case 3
                    obj = obj.fillInFourierGrid_FFT_tri();
                case 4
                    obj = obj.fillInFourierGrid_DFT4();
                case 6
                    obj = obj.fillInFourierGrid_DFT6();
            end
            %class(obj.measuredK)
            obj.measuredK = single(obj.measuredK);
            obj.measuredK_mask = obj.measuredK~=0;
            %obj.measuredK_mask(obj.measuredK==0) = false;
            obj.Vol_ind = My_volumn_index(size(obj.measuredK),[obj.Dim1 obj.Dim2 obj.Dim1]);
        end
        
        function obj = reset_measuredK_mask(obj)
            obj.measuredK_mask = obj.measuredK~=0;
            %obj.measuredK_mask = true(size(obj.measuredK));
            %obj.measuredK_mask(obj.measuredK==0) = false;
        end
        
        % clear big temporary arrays
        function obj = ClearCalcVariables(obj)
            obj.InputProjections=[];
            obj.InputAngles=[];
            %obj.measuredK=[];
            %obj.measuredK_mask=[];
            %         obj.Support=[];
            obj.initialObject=[];
            obj.recIFFT=[];
        end
        
        
        function SaveResults(obj)
            save(obj.filename_Results, 'obj')
        end
        
        
        % set parameters for GENFIRE class
        function obj=set_parameters(obj,varargin)
            if mod(length(varargin),2) ~= 0
                error('GENFIRE: Additional argument list not divisible by 2. Options should be ''key'',''value'' pairs.')
            end
            
            % Apply user-provided options
            par_number = 1;
            while par_number < length(varargin)
                if isprop(obj,varargin{par_number})
                    obj.(varargin{par_number}) = varargin{par_number+1};
                else
                    error('GENFIRE: Invalid option %s provided.',varargin{par_number})
                end
                par_number = par_number + 2;
            end
        end
        
    end
    
end