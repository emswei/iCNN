clear all
format long

dxPix = 0.3 ;
dyPix = 0.4 ;
USF = 2 ;

sigmaPix = 0.15 ;
imageScale = 2.5 ;
CGerror = 5e-3 ;

funcToTest = @(x) ImplicitWienerPadeCG(x, imageScale, sigmaPix, USF, CGerror) ;
inImage = LoadData() ;
inImage = inImage / max(max(inImage)) ;

tic() ;
outImage = CheckVariance(funcToTest, inImage, dxPix, dyPix) ;
toc() ;

figure ; imagesc(outImage) ;
set(gca, 'linewidth', 1, 'fontsize', 11) ;
daspect([1 1]) ;
print('-dpng', 'TestImplicitWienerPadeCGImage') ;

outSpec = fft2(outImage) ;
figure ; imagesc(2*log(abs(outSpec))) ;
set(gca, 'linewidth', 1, 'fontsize', 11) ;
daspect([1 1]) ;
print('-dpng', 'TestImplicitWienerPadeCGSpectrum') ;

[Ny, Nx] = size(outImage) ;
Ly = 1 + floor(50 * USF + 0.5) ;
figure ; plot(1:Nx, outImage(Ly, :)) ; grid ;
print('-dpng', ['TestImplicitWienerPadeCGRow', num2str(Ly)]) ;