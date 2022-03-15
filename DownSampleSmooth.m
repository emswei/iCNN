function [outData] = DownSampleSmooth(inData, rowDSF, colDSF, rowMargin = 0.2, colMargin = 0.2)
  format long

  [nRows, nCols] = size(inData) ;
  newRows = floor(nRows * rowDSF + 0.5) ;
  newCols = floor(nCols * colDSF + 0.5) ;
  halfRows = floor((newRows - 0.5) / 2) ;
  halfCols = floor((newCols - 0.5) / 2) ;
  outData = zeros([newRows, newCols]) ;

  window = RectWindowSmooth(nRows, nCols, rowDSF * (1 - rowMargin), colDSF * (1 - colMargin), 3, 3) ;
  tmpData = fft2(inData) ;
  tmpData = tmpData .* window ;

  for nR = 1 : (halfRows + 1)
    NR = nR ;
    outData(nR, 1:(halfCols+1)) = tmpData(NR, 1:(halfCols+1)) ;
    outData(nR, (newCols-halfCols+1):newCols) = tmpData(NR, (nCols-halfCols+1):nCols) ;
  endfor
  for nR = (newRows - halfRows + 1) : newRows
    NR = nRows - newRows + nR ;
    outData(nR, 1:(halfCols+1)) = tmpData(NR, 1:(halfCols+1)) ;
    outData(nR, (newCols-halfCols+1):newCols) = tmpData(NR, (nCols-halfCols+1):nCols) ;
  endfor

  outData = ifft2(outData) ;
  outData = real(outData) ;
  outData = outData * (newRows / nRows * newCols / nCols) ;
endfunction