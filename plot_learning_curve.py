import matplotlib.pyplot as plt
import pickle
import numpy as np

# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'


# with open("accuracy/1_acc_train_celeba.txt", "rb") as f:
#     accs_train = pickle.load(f)
# with open("accuracy/1_acc_train_celeba_tra.txt", "rb") as f:
#     accs_train_tra = pickle.load(f)
# with open("accuracy/1_acc_train_celeba_dcgan.txt", "rb") as f:
#     accs_train_dcgan = pickle.load(f)
# with open("accuracy/1_acc_train_celeba_cdcgan.txt", "rb") as f:
#     accs_train_cdcgan = pickle.load(f)
# with open("accuracy/1_acc_train_celeba_cyclegan.txt", "rb") as f:
#     accs_train_cyclegan = pickle.load(f)
# with open("accuracy/1_acc_train_celeba_stargan.txt", "rb") as f:
#     accs_train_stargan = pickle.load(f)

with open("accuracy/2_acc_val_celeba.txt", "rb") as f:
    accs_val = pickle.load(f)
with open("accuracy/2_acc_val_celeba_tra.txt", "rb") as f:
    accs_val_tra = pickle.load(f)
with open("accuracy/2_acc_val_celeba_dcgan.txt", "rb") as f:
    accs_val_dcgan = pickle.load(f)
with open("accuracy/2_acc_val_celeba_cdcgan.txt", "rb") as f:
    accs_val_cdcgan = pickle.load(f)
with open("accuracy/2_acc_val_celeba_cyclegan.txt", "rb") as f:
    accs_val_cyclegan = pickle.load(f)
with open("accuracy/2_acc_val_celeba_stargan.txt", "rb") as f:
    accs_val_stargan = pickle.load(f)

# with open("accuracy/64_acc_train_celeba_stargan.txt", "rb") as f:
#     accs_train_stargan = pickle.load(f)
# with open("accuracy/64_acc_val_celeba_stargan.txt", "rb") as f:
#     accs_val_stargan = pickle.load(f)

plt.style.use('ggplot')
# plt.plot(accs_train, label='No augmentation')
# plt.plot(accs_train_tra, label='Geometric')
# plt.plot(accs_train_dcgan, label='DCGAN')
# plt.plot(accs_train_cdcgan, label='cDCGAN')
# plt.plot(accs_train_cyclegan, label='CycleGAN')
# plt.plot(accs_train_stargan, label='StarGAN')

plt.plot(accs_val, label='No augmentation')
plt.plot(accs_val_tra, label='Geometric')
plt.plot(accs_val_dcgan, label='DCGAN')
plt.plot(accs_val_cdcgan, label='cDCGAN')
plt.plot(accs_val_cyclegan, label='CycleGAN')
plt.plot(accs_val_stargan, label='StarGAN')


plt.xticks(np.arange(0, 20), np.arange(1, 21))
plt.xlabel('Epochs')
plt.title('Validation accuracy during training')
plt.legend()
plt.savefig('group3.png')
