% Reading data into file
tiffPath = 'SampleData_dualbeads-803-235-245-101.tif';
tiffInfo = imfinfo(tiffPath);
numFrames = numel(tiffInfo);
[height, width] = deal(tiffInfo(1).Height, tiffInfo(1).Width);
meta = zeros(height, width, numFrames);
for k = 1:numFrames
    meta(:,:,k) = imread(tiffPath, k);
end



% K-means clustering 
data = reshape(meta, [], numFrames); % Pixels × wavelengths
numClusters = 3; % Adjust based on expected components
[idx, C] = kmeans(data, numClusters); % C contains cluster centroids (base spectra)
base = C'; % Transpose to match code's input format (wavelengths × components)

[imgfit, conc] = imgcfit_mod(meta, base, 'display', 'on');

fftData = fft(data, [], 2);
G = real(fftData(:,2)); % First harmonic real part
S = imag(fftData(:,2)); % First harmonic imaginary part
figure;
scatter(G, S, 5, idx, 'filled'); % Color by cluster
xlabel('G'); ylabel('S'); title('Phasor Plot');