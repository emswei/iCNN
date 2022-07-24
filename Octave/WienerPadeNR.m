function [outImage] = WienerPadeNR(inImage, imageScale, sigmaPix, USF = 1)
  format long

  if USF > 1
    gaussImage = UpSample(inImage, USF, USF) ;
  endif
  if sigmaPix > 1e-6
    gaussImage = GaussBlur(gaussImage, sigmaPix * USF, sigmaPix * USF) ;
  endif

  gaussImage2 = gaussImage .^ 2 ;
  DI = gaussImage2 + 1 / imageScale^2 ; % denominator image
  ID = NewtonRaphsonDivision(DI, [], 10, 5e-3) ;
  outImage = gaussImage2 .* ID ;

  if 1 % whether to clip off the near-Nyquist spectral components
    [nRows, nCols] = size(outImage) ;
    window = RectWindowSmooth(nRows, nCols, 0.85, 0.85, 3, 3) ;
    outImage = LowPassSmooth(outImage, window) ;
  endif
  outImage = outImage * ((1 + imageScale^2) / imageScale^2) ;
endfunction
