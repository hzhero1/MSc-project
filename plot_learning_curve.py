import matplotlib.pyplot as plt
import pickle

# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'

loss_train_dir = "loss/10_epochs_64_loss_train_celeba.txt"
loss_val_dir = "loss/10_epochs_64_loss_val_celeba.txt"


# with open("loss/10_epochs_64_loss_train_celeba.txt", "rb") as f:
#     losses_val = pickle.load(f)
# with open("loss/10_epochs_64_loss_train_celeba_tra.txt", "rb") as f:
#     losses_val_tra = pickle.load(f)
# with open("loss/10_epochs_64_loss_train_celeba_dcgan.txt", "rb") as f:
#     losses_val_dcgan = pickle.load(f)
# with open("loss/10_epochs_64_loss_train_celeba_cdcgan.txt", "rb") as f:
#     losses_val_cdcgan = pickle.load(f)
# with open("loss/10_epochs_64_loss_train_celeba_cyclegan.txt", "rb") as f:
#     losses_val_cyclegan = pickle.load(f)
# with open("loss/10_epochs_64_loss_train_celeba_stargan.txt", "rb") as f:
#     losses_val_stargan = pickle.load(f)

with open("accuracy/64_acc_train_celeba_stargan.txt", "rb") as f:
    accs_train_stargan = pickle.load(f)
with open("accuracy/64_acc_val_celeba_stargan.txt", "rb") as f:
    accs_val_stargan = pickle.load(f)

plt.style.use('ggplot')
# plt.plot(losses_val, label='No augmentation')
# plt.plot(losses_val_tra, label='Geometric')
# plt.plot(losses_val_dcgan, label='DCGAN')
# plt.plot(losses_val_cdcgan, label='cDCGAN')
# plt.plot(losses_val_cyclegan, label='CycleGAN')
# plt.plot(losses_val_stargan, label='StarGAN')

plt.plot(accs_train_stargan, label='CycleGAN')
plt.plot(accs_val_stargan, label='StarGAN')

plt.xlabel('Epochs')
plt.title('Learning curve')
plt.legend()
plt.savefig('plot_64_10_epochs.png')