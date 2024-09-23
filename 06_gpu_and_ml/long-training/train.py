import lightning as L
from torch import nn, optim, utils
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor


# define the LightningModule
class LitAutoEncoder(L.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def get_autoencoder(checkpoint_path=None):
    # define any number of nn.Modules (or use your current ones)
    print("Defining encoder and decoder")
    encoder = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))
    decoder = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))

    return LitAutoEncoder(encoder, decoder)


def get_train_loader(data_dir):
    # setup data
    print("Setting up data")
    dataset = MNIST(data_dir, download=True, transform=ToTensor())
    train_loader = utils.data.DataLoader(dataset)
    return train_loader
