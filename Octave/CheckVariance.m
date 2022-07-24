function [outImage] = CheckVariance(funcToTest, inImage, dxPix, dyPix)
  format long

  goldImage = feval(funcToTest, inImage) ;
  outImage = ShiftBackForth(funcToTest, inImage, dxPix, dyPix) ;
  RSV = max(max(abs(outImage - goldImage))) / max(max(abs(goldImage)))
endfunction