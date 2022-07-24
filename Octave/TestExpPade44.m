format long

R = 2 ;

x = -R : 0.01 : R ;
yExac = exp(-x) ;
yPade = ExpPade44(x) ;
RelError = 0.5 * max(abs(yPade - yExac) ./ (yPade + yExac))

plot(x, yExac, 'r') ; grid ; hold on ;
plot(x, yPade, 'b') ; hold off ;