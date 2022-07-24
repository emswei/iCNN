function [out] = RectWindow(nRows, nCols, rowPassRatio, colPassRatio)
  format long

  halfRowPass = nRows / 2 * rowPassRatio ;
  halfColPass = nCols / 2 * colPassRatio ;

  out = zeros(nRows, nCols) ;
  for nR = 1 : nRows
    mR = nR - 1 ;
    mR = mR - nRows * (mR >= nRows / 2) ;
    wR = abs(mR) < halfRowPass ;
    for nC = 1 : nCols
      mC = nC - 1 ;
      mC = mC - nCols * (mC >= nCols / 2) ;
      wC = abs(mC) < halfColPass ;
      out(nR, nC) = wR * wC;
    endfor
  endfor

  %figure ; imagesc(out) ;
  %out = circshift(out, [nRows / 2, nCols / 2]) ;
  %figure ; imagesc(out) ;
endfunction