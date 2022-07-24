function [outData] = UpSample(inData, rowUSF, colUSF)
  format long

  [nRows, nCols] = size(inData) ;
  halfRows = floor((nRows - 0.5) / 2) ;
  halfCols = floor((nCols - 0.5) / 2) ;
  newRows = floor(nRows * rowUSF + 0.5) ;
  newCols = floor(nCols * colUSF + 0.5) ;
  outData = zeros([newRows, newCols]) ;

  tmpData = fft2(inData) ;

  for nR = 1 : (halfRows + 1)
    NR = nR ;
    outData(NR, 1:(halfCols+1)) = tmpData(nR, 1:(halfCols+1)) ;
    outData(NR, (newCols-halfCols+1):newCols) = tmpData(nR, (nCols-halfCols+1):nCols) ;
  endfor
  for nR = (nRows - halfRows + 1) : nRows
    NR = newRows - nRows + nR ;
    outData(NR, 1:(halfCols+1)) = tmpData(nR, 1:(halfCols+1)) ;
    outData(NR, (newCols-halfCols+1):newCols) = tmpData(nR, (nCols-halfCols+1):nCols) ;
  endfor

  outData = ifft2(outData) ;
  outData = real(outData) ;
  outData = outData * (newRows / nRows * newCols / nCols) ;
endfunction