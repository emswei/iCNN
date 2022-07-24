clear all
format long

dxPix = 0.3 ;
dyPix = 0.4 ;

funcToTest = @(x) sin((pi / 2) * x) ;
inImage = LoadData() ;
outImage = CheckVariance(funcToTest, inImage, dxPix, dyPix) ;

figure ; imagesc(outImage) ;
set(gca, 'linewidth', 1, 'fontsize', 11) ;
daspect([1 1]) ;
print('-dpng', 'TestSinImage') ;

outSpec = fft2(outImage) ;
figure ; imagesc(2*log(abs(outSpec))) ;
set(gca, 'linewidth', 1, 'fontsize', 11) ;
daspect([1 1]) ;
print('-dpng', 'TestSinSpectrum') ;

[Ny, Nx] = size(outImage) ;
Ly = 1 + floor(50 * 1 + 0.5) ;
figure ; plot(1:Nx, outImage(Ly, :)) ; grid ;
print('-dpng', ['TestSinRow', num2str(Ly)]) ;