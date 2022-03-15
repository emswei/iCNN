function [outData] = LowPassSmooth(inData, window)
  format long

  outData = fft2(inData) ;
  outData = outData .* window ;
  outData = ifft2(outData) ;
  outData = real(outData) ;
endfunction