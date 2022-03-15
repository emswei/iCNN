clear all
format long

dxPix = 0.5 ;
dyPix = 0.3 ;
USF = 2 ;

sigmaPix = 0.15 ;
imageScale = 2.5 ;

funcToTest = @(x) WienerPadeNR(x, imageScale, sigmaPix, USF) ;
inImage = LoadData() ;
inImage = inImage / max(max(inImage)) ;

tic() ;
outImage = CheckVariance(funcToTest, inImage, dxPix, dyPix) ;
toc() ;

outSpec = fft2(outImage) ;
figure ; imagesc(2*log(abs(outSpec))) ;
set(gca, 'linewidth', 1, 'fontsize', 11) ;
daspect([1 1]) ;
print('-dpng', ['TestWienerPadeNRUp', num2str(USF), 'Spectrum']) ;

figure ; imagesc(outImage) ;
set(gca, 'linewidth', 1, 'fontsize', 11) ;
daspect([1 1]) ;
print('-dpng', ['TestWienerPadeNRUp', num2str(USF), 'Image']) ;

[Ny, Nx] = size(outImage) ;
Ly = 1 + floor(50 * USF + 0.5) ;
figure ; plot(1:Nx, outImage(Ly, :)) ; grid ;
print('-dpng', ['TestWienerPadeNRUp', num2str(USF), 'Row', num2str(Ly)]) ;
