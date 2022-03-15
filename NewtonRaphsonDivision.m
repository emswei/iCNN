% The Newton-Raphson division is employed to compute an approximation
% for y = 1 / x ; A spectral window is for optional spectrum control.
function [y] = NewtonRaphsonDivision(xIn, window = [], maxIters = 10, epsilon = 0)
  format long

  if ! isempty(window)
    x = LowPassSmooth(xIn, window) ;
  else
    x = xIn ;
  endif
  y = ones(size(x)) / 2 ;
  NRerrorOld = Inf ;

  for iter = 0 : maxIters
    ErrCorr = 1 - x .* y ;
    if ! isempty(window)
      ErrCorr = LowPassSmooth(ErrCorr, window) ;
    endif
    NRerror = max(max(abs(ErrCorr))) ;
    display(['iter ', num2str(iter), ', NRerror = ', num2str(NRerror)]) ;

    if NRerror < epsilon
      disp(['NR has reached accuracy goal']) ;
      break ;
    endif
    if iter > 2 && NRerror > NRerrorOld * (1 - 1 / maxIters)
      disp(['NR has reached an optimum']) ;
      break ;
    endif

    y = y + y .* ErrCorr ;
    if ! isempty(window)
      y = LowPassSmooth(y, window) ;
    endif
    NRerrorOld = NRerror ;
  endfor
endfunction