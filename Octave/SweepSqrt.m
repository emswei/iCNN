clear all
format long

xSteps = 5 ;
ySteps = 5 ;
inUSF = 2 ;

funcToTest = @sqrt ;
inImage = UpSample(LoadData(), inUSF, inUSF) ;
outImage = SweepVariance(funcToTest, inImage, xSteps, ySteps) ;