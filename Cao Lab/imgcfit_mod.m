% This code use the approach described in the reference to realize image
% reconstruction based on known spectra. 
% Hyperspectral Imaging with Stimulated Raman Scattering by Chirped Femtosecond Lasers
% Dan Fu, Gary Holtom, Christian Freudiger, Xu Zhang, and Xiaoliang Sunney Xie
% The Journal of Physical Chemistry B 2013 117 (16), 4634-4640
% DOI: 10.1021/jp308938t


function [imgfit,conc] = imgcfit_mod(meta,base,varargin)
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic;
dispop = 'on';
fitmode = 'multicore';
% reading optional parameters~
if (rem(length(varargin),2) == 1)
    error('Optional parameters come in pairs!');
elseif ~isempty(varargin)
    for i = 1:2:length(varargin)-1
        if ~ischar(varargin{i})
            error('Unknown optional parameter name type, name must be string!');
        end
        % change value of parameter
        switch lower(varargin{i})
            case 'fitmode'
                fitmode = lower(varargin{i+1});
            case 'display'
                dispop = lower(varargin{i+1});

        end
    end
end

% prepare spectra for fitting
numofIC = min(size(base));
for i=1:1:numofIC
%    base(:,i) = base(:,i)-min(base(:,i));
    base(:,i) = base(:,i)/max(base(:,i));
end

dim = size(meta);
pidx = dim(1)*dim(2);
% preparing parameters used in fitting
conc = zeros(pidx,numofIC);
conc = conc';
tolx = 20*eps*norm(base,1)*length(base);
options = optimset('TolX',tolx);

% transform 3D stack into 2D
spec = double(reshape(meta,pidx,dim(3))); 
spec = spec'; % column vectors are spectra


disp('Preparation for Fitting done!');
toc;
disp('=================================================');

% reconstruct image by fitting
if strcmp(fitmode,'singlecore')
    for i = 1:1:pidx
            specmed = spec(:,i);
            conc(:,i) = lsqnonneg(base,specmed,options);
    end
else
    parfor i = 1:pidx
            specmed = spec(:,i);
            conc(:,i) = lsqnonneg(base,specmed,options);
    end
end
imgfit = reshape(conc',dim(1),dim(2),numofIC);

disp('Fitting reconstruction Done!');
toc;
disp('=================================================');

% for display, default to display, can be turned off by given 'off' value
% to dispop
if strcmp(dispop,'on')
    imgdis = zeros(dim(1),dim(2),numofIC);
    imgdisf = zeros(dim(1),dim(2),numofIC);
    
    for i = 1:1:numofIC
        imgdis(:,:,i) = imgfit(:,:,i)/max(max(imgfit(:,:,i)));
        imgdisf(:,:,i) = imgfit(:,:,i)/max(max(imgfit(:,:,i)));
    end
    if numofIC < 3
        imgdis(:,:,numofIC+1) = uint8(zeros(dim(1),dim(2)));
        imgdisf(:,:,numofIC+1) = uint8(zeros(dim(1),dim(2)));
    end
    imgdis = uint8(imgdis*255);
    imgdisf = uint8(imgdisf*255);
    if numofIC <= 3
        figure('Name','Reconstructed Image by Fitting');
        subplot(1,2,1),imshow(imgdis);title('raw fitting');
        subplot(1,2,2),imshow(imgdisf);title('filtered fitting');
    end
end
end
