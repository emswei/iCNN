function [data] = LoadData()
  format long

  DATA = load('OrigData.txt') ;
  data = DownSampleSmooth(DATA, 1/2, 1/2) ;

  %save('-ascii', 'TestData.txt', 'data') ;
endfunction