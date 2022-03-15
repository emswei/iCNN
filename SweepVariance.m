function [outImage] = SweepVariance(funcToTest, inImage, xSteps, ySteps)
  format long

  goldImage = feval(funcToTest, inImage) ;
  outImage = zeros(size(goldImage)) ;

  maxVariance = 0 ;
  for iy = 0 : (ySteps - 1)
    dyPix = iy / ySteps ;
    for ix = 0 : (xSteps - 1)
      dxPix = ix / xSteps ;
      if ix == 0 && iy == 0
        break ;
      endif
      disp(['[dxPix, dyPix] = [', num2str(dxPix), ', ', num2str(dyPix), ']']) ;
      thisImage = ShiftBackForth(funcToTest, inImage, dxPix, dyPix) ;
      thisVariance = max(max(abs(thisImage - goldImage))) / max(max(abs(goldImage)))
      if maxVariance < thisVariance
        maxVariance = thisVariance ;
        outImage = thisImage ;
      endif
    endfor
  endfor
  MaxRelativeVariance = maxVariance
endfunction