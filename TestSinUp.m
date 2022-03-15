clear all
format long

dxPix = 0.3 ;
dyPix = 0.4 ;
inUSF = 2 ;

funcToTest = @(x) sin((pi / 2) * x) ;
inImage = UpSample(LoadData(), inUSF, inUSF) ;
outImage = CheckVariance(funcToTest, inImage, dxPix, dyPix) ;

figure ; imagesc(outImage) ;
set(gca, 'linewidth', 1, 'fontsize', 11) ;
daspect([1 1]) ;
print('-dpng', 'TestSinUpImage') ;

outSpec = fft2(outImage) ;
figure ; imagesc(2*log(abs(outSpec))) ;
set(gca, 'linewidth', 1, 'fontsize', 11) ;
daspect([1 1]) ;
print('-dpng', 'TestSinUpSpectrum') ;

[Ny, Nx] = size(outImage) ;
Ly = 1 + floor(50 * inUSF + 0.5) ;
figure ; plot(1:Nx, outImage(Ly, :)) ; grid ;
print('-dpng', ['TestSinUpRow', num2str(Ly)]) ;