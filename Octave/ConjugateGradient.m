% This implementation is adapted from the sample code in the Wiki page
% https://en.wikipedia.org/wiki/Conjugate_gradient_method. The argument
% "A" is a function handle or name that realizes a Hermitian operator,
% "R" is an optional function handle or name that regulates the result.
function [x] = ConjugateGradient(A, b, R = [], x = [], maxIters = 1000, epsilon = 0)
  format long

  if isempty(x)
    x = zeros(size(b)) ;
    r = zeros(size(b)) ;
  elseif ! isempty(R)
    x = feval(R, x) ;
    r = feval(A, x) ;
  endif

  r = b - r ;
  p = r ;
  rsold = r' * r ;
  rs0 = rsold ;

  if sqrt(rs0 / max(b' * b, x' * x)) < epsilon
    return ;
  endif

  for i = 1 : max(maxIters, length(b))
    Ap = feval(A, p) ;
    if ! isempty(R)
      Ap = feval(R, Ap) ;
    endif
    alpha = rsold / (p' * Ap) ;
    r = r - alpha * Ap ;
    rsnew = r' * r ;
    disp(['CG iteration ', num2str(i), ', residue = ', num2str(sqrt(rsnew/rs0))]) ;

    if sqrt(rsnew / rs0) < epsilon
      disp(['CG has reached accuracy goal']) ;
      break ;
    endif
    if rsnew >= rsold
      disp(['CG has reached an optimum']) ;
      break ;
    endif

    x = x + alpha * p ;
    p = r + (rsnew / rsold) * p ;
    rsold = rsnew ;
  endfor
endfunction