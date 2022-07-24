clear all
format long

data = LoadData() ;
figure ; imagesc(data) ;
set(gca, 'linewidth', 1, 'fontsize', 11) ;
daspect([1 1]) ;
print('-dpng', 'OrigImage') ;

spec = fft2(data) ;
figure ; imagesc(2*log(abs(spec))) ;
set(gca, 'linewidth', 1, 'fontsize', 11) ;
daspect([1 1]) ;
print('-dpng', 'OrigSpectrum') ;