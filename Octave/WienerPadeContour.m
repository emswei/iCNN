format long

USF = 8 ;
threshold = 0.45 ;

sigmaPix = 0.15;
imageScale = 2.5 ;

dX = 0.05 ; % [um]
dY = 0.05 ; % [um]

data = LoadData() ;
DATA = WienerPade(data, imageScale, sigmaPix, USF) ;

X = (0 : (floor(128 * USF + 0.5) - 1)) * dX / USF ;
Y = (0 : (floor(128 * USF + 0.5) - 1)) * dY / USF ;
figure ; contour(X, Y, DATA, [threshold, threshold], 'b', 'linewidth', 1) ;

set(gca, 'linewidth', 1, 'fontsize', 12) ;
set(gca, 'xlabel', 'x [{\mu}m]', 'fontsize', 12) ;
set(gca, 'ylabel', 'y [{\mu}m]', 'fontsize', 12) ;
daspect([1 1]) ;
print('-dpng', 'WienerPadeContour') ;