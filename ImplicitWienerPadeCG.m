function [outImage] = ImplicitWienerPadeCG(inImage, imageScale, sigmaPix, USF, CGerror)
  format long

  if USF > 1
    gaussImage = UpSample(inImage, USF, USF) ;
  endif
  if sigmaPix > 1e-6
    gaussImage = GaussBlur(gaussImage, sigmaPix * USF, sigmaPix * USF) ;
  endif

  allUSF = USF * 5 ;
  gaussImage2 = UpSample(gaussImage, allUSF/USF, allUSF/USF) ;
  gaussImage2 = gaussImage2 .^ 2 ;
  [nRows, nCols] = size(gaussImage2) ;

  NI = gaussImage2 ; % numerator image
  DI = gaussImage2 + 1 / imageScale^2 ; % denominator image
  A = 2 * conj(DI(:)) .* DI(:) ; % as a 1D vector
  b = 2 * DI(:) .* NI(:) ; % as a 1D vector
  window = RectWindowSmooth(nRows, nCols, USF/allUSF, USF/allUSF, 3, 3) ;

  functionA = @(v) A .* v ;
  functionR = @(v) LowPassRegulator(v, nRows, nCols, window) ;
  y = ConjugateGradient(functionA, b, functionR, [], 100, CGerror) ;

  outImage = reshape(y, nRows, nCols) ;
  outImage = DownSampleSmooth(outImage, USF/allUSF, USF/allUSF, 0.15, 0.15) ;
  outImage = outImage * ((1 + imageScale^2) / imageScale^2) ;
endfunction