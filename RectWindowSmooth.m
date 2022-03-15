function [out] = RectWindowSmooth(nRows, nCols, rowPassRatio, colPassRatio, rowSigma, colSigma)
  format long

  out = RectWindow(nRows, nCols, rowPassRatio, colPassRatio) ;
  out = GaussBlur(out, rowSigma, colSigma) ;

  %figure ; imagesc(out) ;
  %out = circshift(out, [nRows / 2, nCols / 2]) ;
  %figure ; imagesc(out) ;
endfunction