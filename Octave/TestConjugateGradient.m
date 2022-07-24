clear all
format long

M = [1.1, 0.3, 0.5; 0.3, 1.5, 0.7; 0.5, 0.7, 1.9]
x0 = [1, -1, 1]'
y0 = M * x0

A = 2 * M' * M ;
b = 2 * M * y0 ;

disp(['Now use ConjugateGradient() to find an x that minimizes ||M * x - y0||^2 :']) ;
disp(['']) ;
disp(['The corresponding positive-definite matrix 2 * M'' * M = ']) ; A
disp(['The corresponding colum vector 2 * M * y0 =']) ; b

functionA = @(v) A * v ;
x = ConjugateGradient(functionA, b, [], [], 1000, 1e-12)