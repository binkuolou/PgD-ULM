clear; clc;

% Load required data

datapath = 'The PSF path';
savepath = 'The Training datapath\';

labelPath = [savepath, 'label\']; % The 8x unsampled localization map
inputPath = [savepath, 'input\']; % The MBs map with 1lambda pixel size
posPath   = [savepath, 'pos\'];  % The amp and the postion information for Decode based model

mkdir(labelPath); mkdir(inputPath); mkdir(posPath);

% Parameters
numData = 10000; % The number of training datasets generated 
x = 64;
X = x * 8;
pixel_size = 0.1; % mm
Concentration_bubble = [0.8, 3.2] * pixel_size^2; % definition of the concentration. 
N = round(Concentration_bubble * x^2);

interpF = X / x;

% PSF loading setup
indexPSF = 1;
index = 1;
load([datapath, 'PSF_DDIM_w0_0_10.mat']);
indexPSF = indexPSF + 1;

for nn = 1:numData
    fprintf('%d\n', nn);
    
    inputData = zeros(x, x);
    labelData = zeros(X, X);
    inputData_origin = zeros(X, X);
    
    % Get bubble positions
    n = randi([N(1) N(2)]);
    
    bubbleX = round((rand(n,1))*X);
    bubbleY = round((rand(n,1))*Y);
    
    % Apply SNR-based amplitude threshold
    SNR = -10 - 20 * rand;
    maxAmp = max(bubbleAmp(:));
    bubbleAmpThresh = maxAmp * 10^(SNR/20);
    bubbleAmp(bubbleAmp < bubbleAmpThresh) = bubbleAmp(bubbleAmp < bubbleAmpThresh) + bubbleAmpThresh;
    
    gt_pts = [];
    i = 1;
    
    while i <= n
        pointPSF = squeeze(PSF(index, :, :, :));
        pointIndex = bubblePos(i, :);
        pointAmp = bubbleAmp(i);
        index = index + 1;
        
        % Normalize PSF
        pointPSF = (pointPSF - min(pointPSF(:))) / (max(pointPSF(:)) - min(pointPSF(:)));
        
        % Zero out central lines to avoid artifacts
        pointPSF(:, 49) = 0;
        pointPSF(49, :) = 0;
        
        % Reload PSF if exhausted
        if index > 512
            load([datapath, 'PSF_DDIM_w0_', num2str(indexPSF), '_10.mat']);
            index = 1;
            indexPSF = indexPSF + 1;
        end
        
        % Create delta and convolve with PSF
        Point = zeros(size(pointPSF));
        Point(25, 25) = pointAmp;
        Point = conv2(Point, pointPSF, 'same');
        
        % Use cropped PSF for localization
        pointPSF_local = pointPSF(9:40, 9:40);
        [Zc, Xc, ~] = LocRadialSym(pointPSF_local, 17, 17);
        
        % Skip if localization is too far off-center
        if abs(Zc) > 8.5 || abs(Xc) > 8.5
            i = i + 1;
            continue;
        end
        
        pointIndex_origin = (pointIndex - 1)/8 + 1 + [Zc, Xc];
        pointIndex_label = round(pointIndex + [Zc, Xc]);
        
        % Check bounds for label
        inBoundsLabel = all(pointIndex_label >= 1) && all(pointIndex_label <= [X, X]);
        
        % Define PSF support
        psfVector = -24:24;
        px = length(psfVector);
        
        % Check bounds for full convolution
        margin = floor(px/2);
        inBoundsFull = all(pointIndex > margin) && all(pointIndex <= [X - margin, X - margin]);
        
        if inBoundsLabel
            labelData(pointIndex_label(1), pointIndex_label(2)) = ...
                labelData(pointIndex_label(1), pointIndex_label(2)) + 1;
            
            if inBoundsFull
                inputData_origin(pointIndex(1)+psfVector, pointIndex(2)+psfVector) = ...
                    inputData_origin(pointIndex(1)+psfVector, pointIndex(2)+psfVector) + Point;
            else
                % Pad temporarily if near edge
                temp = padarray(inputData_origin, [margin, margin], 0);
                idx_pad = pointIndex + margin;
                temp(idx_pad(1)+psfVector, idx_pad(2)+psfVector) = ...
                    temp(idx_pad(1)+psfVector, idx_pad(2)+psfVector) + Point;
                inputData_origin = temp(margin+1:end-margin, margin+1:end-margin);
            end
        end
        
        gt_pts = [gt_pts; pointIndex_origin];
        i = i + 1;
    end
    
    % Downsample and add noise
    inputData_origin = conv2(inputData_origin, ones(8)/64, 'same');
    inputData = inputData_origin(1:interpF:end, 1:interpF:end);
    
    NoiseParam.Power        = -2;      % [dBW]
    NoiseParam.Impedance    = 0.2;     % [ohms]
    NoiseParam.SigmaGauss   = 1.5;
    NoiseParam.clutterdB    = SNR;
    NoiseParam.amplCullerdB = 10;
    inputData = PALA_AddNoiseInIQ(inputData, NoiseParam);
    
    % Save
    save([inputPath, 'Input', num2str(nn), '.mat'], 'inputData');
    save([labelPath, 'Label', num2str(nn), '.mat'], 'labelData');
    save([posPath,   'GT_Pos', num2str(nn), '.mat'], 'gt_pts');
end

%% Helper Functions (unchanged)
function [Zc,Xc,sigma] = LocRadialSym(Iin,fwhm_z,fwhm_x)
    [Zc,Xc] = localizeRadialSymmetry(Iin,fwhm_z,fwhm_x);
    sigma = ComputeSigmaScat(Iin,Zc,Xc);
end

function sigma = ComputeSigmaScat(Iin,Zc,Xc)
    [Nz,Nx] = size(Iin);
    Isub = Iin - mean(Iin(:));
    [pz,px] = meshgrid(1:Nx, 1:Nz);
    zoffset = pz - Zc + Nz/2;
    xoffset = px - Xc + Nx/2;
    r2 = zoffset.^2 + xoffset.^2;
    sigma = sqrt(sum(sum(Isub .* r2)) / sum(Isub(:))) / 2;
end