function [outImage] = ShiftBackForth(funcToTest, testImage, dxPix, dyPix)
  format long

  inImage = testImage ;
  [inNy, inNx] = size(inImage) ;
  inImage = ShiftX(inImage, dxPix) ;
  inImage = ShiftY(inImage, dyPix) ;

  outImage = feval(funcToTest, inImage) ;
  [outNy, outNx] = size(outImage) ;
  outImage = ShiftY(outImage, (-dyPix) * outNy / inNy) ;
  outImage = ShiftX(outImage, (-dxPix) * outNx / inNx) ;
endfunction