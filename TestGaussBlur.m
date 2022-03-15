clear all
format long

dxPix = 0.3 ;
dyPix = 0.4 ;
rowSigma = 2.0 ;
colSigma = 2.0 ;

funcToTest = @(x) GaussBlur(x, rowSigma, colSigma) ;
inImage = LoadData() ;
outImage = CheckVariance(funcToTest, inImage, dxPix, dyPix) ;

figure ; imagesc(outImage) ;
set(gca, 'linewidth', 1, 'fontsize', 11) ;
daspect([1 1]) ;
print('-dpng', 'GaussBlurImage') ;

outSpec = fft2(outImage) ;
figure ; imagesc(2*log(abs(outSpec))) ;
set(gca, 'linewidth', 1, 'fontsize', 11) ;
daspect([1 1]) ;
print('-dpng', 'GaussBlurSpectrum') ;

[Ny, Nx] = size(outImage) ;
Ly = 1 + floor(50 * 1 + 0.5) ;
figure ; plot(1:Nx, outImage(Ly, :)) ; grid ;
print('-dpng', ['GaussBlurRow', num2str(Ly)]) ;