import matplotlib.pyplot as plt
import pickle

# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'

loss_train_dir = "loss/loss_train.txt"
loss_val_dir = "loss/loss_val.txt"

with open(loss_train_dir, "rb") as f:
    losses_train = pickle.load(f)
with open(loss_val_dir, "rb") as f:
    losses_val = pickle.load(f)

plt.style.use('ggplot')
plt.plot(losses_train, label='Training loss')
plt.plot(losses_val, label='Validation loss')
plt.xlabel('Epochs')
plt.title('Learning curve')
plt.legend()
plt.savefig('plot.png')