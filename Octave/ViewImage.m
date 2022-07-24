function [] = ViewImage(sFileNameOfOneDimDataArray)
  format long

  data = load(sFileNameOfOneDimDataArray) ;
  [nRows, nCols] = size(data)

  figure ; imagesc(data) ;
  set(gca, 'YDir', 'normal') ;
endfunction