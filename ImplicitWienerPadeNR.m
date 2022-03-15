function [outImage] = ImplicitWienerPadeNR(inImage, imageScale, sigmaPix, USF, window, NRerror)
  format long

  if USF > 1
    gaussImage = UpSample(inImage, USF, USF) ;
  endif
  if sigmaPix > 1e-6
    gaussImage = GaussBlur(gaussImage, sigmaPix * USF, sigmaPix * USF) ;
  endif

  gaussImage2 = UpSample(gaussImage, 2, 2) ;
  gaussImage2 = gaussImage2 .^ 2 ;

  DI = gaussImage2 + 1 / imageScale^2 ; % denominator image
  ID = NewtonRaphsonDivision(DI, window, 10, NRerror) ;
  outImage = LowPassSmooth(gaussImage2, window) .* ID ;

  outImage = DownSampleSmooth(outImage, 1/2, 1/2, 0.15, 0.15) ;
  outImage = outImage * ((1 + imageScale^2) / imageScale^2) ;
endfunction