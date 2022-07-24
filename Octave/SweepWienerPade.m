clear all
format long

xSteps = 5 ;
ySteps = 5 ;
inUSF = 2 ;

sigmaPix = 0.15 * inUSF ;
imageScale = 2.0 ;

funcToTest = @(x) WienerPade(x, sigmaPix, imageScale) ;
inImage = UpSample(LoadData(), inUSF, inUSF) ;
outImage = SweepVariance(funcToTest, inImage, xSteps, ySteps) ;