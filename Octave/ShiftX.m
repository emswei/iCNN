function [outData] = ShiftX(inData, dxPix)
  format long

  [Ny, Nx] = size(inData) ;
  Ns = Nx ;
  Ds = dxPix ;

  outSpec = fft2(inData) ;

  for n = 1 : Ns
    m = n - 1 ;
    m = m - (m >= Ns / 2) * Ns ;
    phasor = exp(-j * 2 * pi * m * Ds / Ns) ;
    outSpec(:, n) = outSpec(:, n) * phasor ;
  endfor

  outData = ifft2(outSpec) ;
  outData = real(outData) ;
endfunction