clear all
format long

dxPix = 0.5 ;
dyPix = 0.3 ;
USF = 2 ;

sigmaPix = 0.15 ;
imageScale = 2.5 ;
NRerror = 5e-3 ;

inImage = LoadData() ;
inImage = inImage / max(max(inImage)) ;

dims = 2 * floor(USF * size(inImage) + 0.5) ;
window = RectWindowSmooth(dims(1), dims(2), 0.4, 0.4, 3, 3) ;
funcToTest = @(x) ImplicitWienerPadeNR(x, imageScale, sigmaPix, USF, window, NRerror) ;

tic() ;
outImage = CheckVariance(funcToTest, inImage, dxPix, dyPix) ;
toc() ;

figure ; imagesc(outImage) ;
set(gca, 'linewidth', 1, 'fontsize', 11) ;
daspect([1 1]) ;
print('-dpng', 'TestImplicitWienerPadeNRImage') ;

outSpec = fft2(outImage) ;
figure ; imagesc(2*log(abs(outSpec))) ;
set(gca, 'linewidth', 1, 'fontsize', 11) ;
daspect([1 1]) ;
print('-dpng', 'TestImplicitWienerPadeNRSpectrum') ;

[Ny, Nx] = size(outImage) ;
Ly = 1 + floor(50 * USF + 0.5) ;
figure ; plot(1:Nx, outImage(Ly, :)) ; grid ;
print('-dpng', ['TestImplicitWienerPadeNRRow', num2str(Ly)]) ;