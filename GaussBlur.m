% This function blurs or smoothes inImage by convolving it with a normalized
% Gaussian point spread function with rowSigma, colSigma in units of pixels.
function [outImage] = GaussBlur(inImage, rowSigma, colSigma)
  format long
  [nRows, nCols] = size(inImage) ;

  outImage = fft2(inImage) ;
  sigmaRow = nRows / (2 * pi) / rowSigma ; % conjugate sigmas in frequency
  sigmaCol = nCols / (2 * pi) / colSigma ; % conjugate sigmas in frequency
  for nR = 1 : nRows
    mR = nR - 1 ;
    mR = mR - nRows * (mR >= nRows / 2) ;
    gaussRow = exp(-(mR / sigmaRow)^2 / 2) ;
    for nC = 1 : nCols
      mC = nC - 1 ;
      mC = mC - nCols * (mC >= nCols / 2) ;
      gaussCol = exp(-(mC / sigmaCol)^2 / 2) ;
      outImage(nR, nC) = outImage(nR, nC) * gaussRow * gaussCol ;
    endfor
  endfor
  outImage = ifft2(outImage) ;
  outImage = real(outImage) ;
endfunction
