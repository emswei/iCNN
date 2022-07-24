% [4/4] Pade approximant for exp(-x)
function [y] = ExpPade44(x)
  format long

  P = 1680 - 840 * x + 180 * x.^2 - 20 * x.^3 + x.^4 ;
  Q = 1680 + 840 * x + 180 * x.^2 + 20 * x.^3 + x.^4 ;
  y = P ./ Q ;
endfunction