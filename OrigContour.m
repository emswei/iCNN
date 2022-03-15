clear all
format long

inUSF = 8 ;
threshold = 0.33 ;

dX = 0.05 ; % [um]
dY = 0.05 ; % [um]

DATA = UpSample(LoadData(), inUSF, inUSF) ;

X = (0 : (floor(128 * inUSF + 0.5) - 1)) * dX / inUSF ;
Y = (0 : (floor(128 * inUSF + 0.5) - 1)) * dY / inUSF ;
figure ; contour(X, Y, DATA, [threshold, threshold], 'b', 'linewidth', 1) ;

set(gca, 'linewidth', 1, 'fontsize', 12) ;
set(gca, 'xlabel', 'x [{\mu}m]', 'fontsize', 12) ;
set(gca, 'ylabel', 'y [{\mu}m]', 'fontsize', 12) ;
daspect([1 1]) ;
print('-dpng', 'OrigContour') ;