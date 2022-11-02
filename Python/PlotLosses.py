import matplotlib.pyplot as plt
from numpy import loadtxt

loss_list = loadtxt('Loss.txt')
val_loss_list = loadtxt('ValLoss.txt')

plt.plot(range(1, 1+len(loss_list)), loss_list, label='loss')
plt.plot(range(1, 1+len(loss_list)), val_loss_list, label='val_loss')
plt.legend(fontsize=14)
plt.xlabel('epochs', fontsize=14)
plt.ylabel('MSE losses', fontsize=14)
plt.grid()
plt.savefig('Losses.png')
plt.show()