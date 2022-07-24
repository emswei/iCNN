function [outVector] = LowPassRegulator(inVector, nRows, nCols, window)
  format long

  inImage = reshape(inVector, nRows, nCols) ; % convert to 2D image
  outImage = LowPassSmooth(inImage, window) ;
  outVector = outImage(:) ; % convert to 1D vector
endfunction