clear all
format long

dxPix = 0.3 ;
dyPix = 0.4 ;
inUSF = 3 ;

funcToTest = @(x) x .^ 3 ;
inImage = UpSample(LoadData(), inUSF, inUSF) ;
outImage = CheckVariance(funcToTest, inImage, dxPix, dyPix) ;

figure ; imagesc(outImage) ;
set(gca, 'linewidth', 1, 'fontsize', 11) ;
daspect([1 1]) ;
print('-dpng', 'TestCubeUpImage') ;

outSpec = fft2(outImage) ;
figure ; imagesc(2*log(abs(outSpec))) ;
set(gca, 'linewidth', 1, 'fontsize', 11) ;
daspect([1 1]) ;
print('-dpng', 'TestCubeUpSpectrum') ;

[Ny, Nx] = size(outImage) ;
Ly = 1 + floor(50 * inUSF + 0.5) ;
figure ; plot(1:Nx, outImage(Ly, :)) ; grid ;
print('-dpng', ['TestCubeUpRow', num2str(Ly)]) ;