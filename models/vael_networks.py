import torch.nn
from torch import nn, split


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, input, hidden_channels, dim):
        return input.reshape(input.size(0), hidden_channels, dim[0], dim[1])


class MNISTPairsEncoder(nn.Module):

    def __init__(self, img_channels=1, hidden_channels=32, latent_dim=8, dropout=0.5):
        super(MNISTPairsEncoder, self).__init__()

        self.img_channels = img_channels
        self.hidden_channels = hidden_channels
        self.latent_dim = latent_dim
        self.label_dim = 20
        self.unflatten_dim = (3, 7)

        self.enc_block_1 = nn.Conv2d(
            in_channels=self.img_channels,
            out_channels=self.hidden_channels,
            kernel_size=4,
            stride=2,
            padding=1)  # hidden_channels x 14 x 28

        self.enc_block_2 = nn.Conv2d(
            in_channels=self.hidden_channels,
            out_channels=self.hidden_channels * 2,
            kernel_size=4,
            stride=2,
            padding=1)  # 2*hidden_channels x 7 x 14

        self.enc_block_3 = nn.Conv2d(
            in_channels=self.hidden_channels * 2,
            out_channels=self.hidden_channels * 4,
            kernel_size=4,
            stride=2,
            padding=1)  # 4*hidden_channels x 2 x 7

        self.flatten = Flatten()

        self.dense_mu = nn.Linear(
            in_features=4 * self.hidden_channels * self.unflatten_dim[0] * self.unflatten_dim[1],
            out_features=self.latent_dim)

        self.dense_logvar = nn.Linear(
            in_features=4 * self.hidden_channels * self.unflatten_dim[0] * self.unflatten_dim[1],
            out_features=self.latent_dim)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # MNISTPairsEncoder block 1
        x = self.enc_block_1(x)
        x = nn.ReLU()(x)
        x = self.dropout(x)

        # MNISTPairsEncoder block 2
        x = self.enc_block_2(x)
        x = nn.ReLU()(x)
        x = self.dropout(x)

        # MNISTPairsEncoder block 3
        x = self.enc_block_3(x)
        x = nn.ReLU()(x)

        # mu and logvar
        x = self.flatten(x)  # batch_size, dim1, dim2, dim3 -> batch_size, dim1*dim2*dim3
        mu, logvar = self.dense_mu(x), self.dense_logvar(x)

        return mu, logvar


class MNISTPairsDecoder(nn.Module):

    def __init__(self, img_channels=1, hidden_channels=32, latent_dim=32, label_dim=9, dropout=0.5,
                 **params):
        super(MNISTPairsDecoder, self).__init__()

        self.img_channels = img_channels
        self.hidden_channels = hidden_channels
        self.latent_dim = latent_dim
        self.padding = [(1, 0), (0, 0), (1, 1)]
        self.unflatten_dim = (3, 7)
        self.label_dim = label_dim

        self.dense = nn.Linear(
            in_features=self.latent_dim + self.label_dim,
            out_features=4 * self.hidden_channels * self.unflatten_dim[0] * self.unflatten_dim[1])

        self.unflatten = UnFlatten()

        self.dec_block_1 = nn.ConvTranspose2d(
            in_channels=self.hidden_channels * 4,
            out_channels=self.hidden_channels * 2,
            kernel_size=(5, 4),
            stride=2,
            padding=1)

        self.dec_block_2 = nn.ConvTranspose2d(
            in_channels=self.hidden_channels * 2,
            out_channels=self.hidden_channels,
            kernel_size=4,
            stride=2,
            padding=1)

        self.dec_block_3 = nn.ConvTranspose2d(
            in_channels=self.hidden_channels,
            out_channels=self.img_channels,
            kernel_size=4,
            stride=2,
            padding=1)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # Unflatten Input
        x = self.dense(x)
        x = self.unflatten(x, self.hidden_channels*4, self.unflatten_dim)

        # MNISTPairsDecoder block 1
        x = self.dec_block_1(x)
        x = nn.ReLU()(x)
        x = self.dropout(x)

        # MNISTPairsDecoder block 2
        x = self.dec_block_2(x)
        x = nn.ReLU()(x)
        x = self.dropout(x)

        # MNISTPairsDecoder block 3
        x = self.dec_block_3(x)
        x = torch.nn.Sigmoid()(x)
        return x


class MNISTPairsMLP(nn.Module):
    def __init__(self, in_features=20, n_facts=10, hidden_channels=20):
        super(MNISTPairsMLP, self).__init__()

        self.n_facts = n_facts
        self.hidden_layer = nn.Linear(in_features=in_features, out_features=hidden_channels)
        self.dense = nn.Linear(in_features=hidden_channels, out_features=n_facts)

    def forward(self, x):
        x = self.hidden_layer(x)
        x = nn.ReLU()(x)
        z = self.dense(x)
        z1, z2 = split(z, self.n_facts // 2, dim=1)

        return z1, z2


#########################################
# Network blocks for Mario program task #
#########################################

class MarioEncoder(nn.Module):

    def __init__(self, img_channels=3, hidden_channels=32, latent_dim=8, dropout=0.0):
        super(MarioEncoder, self).__init__()

        self.img_channels = img_channels
        self.hidden_channels = hidden_channels
        self.latent_dim = latent_dim

        self.flatten = Flatten()

        self.enc_block_1 = nn.Conv2d(in_channels=self.img_channels,
                                     out_channels=self.hidden_channels,
                                     kernel_size=5,
                                     stride=2,
                                     padding=1)

        self.enc_block_2 = nn.Conv2d(in_channels=self.hidden_channels,
                                     out_channels=self.hidden_channels * 2,
                                     kernel_size=5,
                                     stride=2,
                                     padding=1)

        self.enc_block_3 = nn.Conv2d(in_channels=self.hidden_channels * 2,
                                     out_channels=self.hidden_channels * 4,
                                     kernel_size=5,
                                     stride=2,
                                     padding=1)

        self.enc_block_4 = nn.Conv2d(in_channels=self.hidden_channels * 4,
                                     out_channels=self.hidden_channels * 8,
                                     kernel_size=5,
                                     stride=2,
                                     padding=1)

        self.flatten = Flatten()

        self.dense_mu = nn.Linear(
            in_features=self.hidden_channels * 8 * 5 * 5,
            out_features=self.latent_dim)

        self.dense_logvar = nn.Linear(
            in_features=self.hidden_channels * 8 * 5 * 5,
            out_features=self.latent_dim)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.enc_block_1(x)  # 49,49
        x = nn.SELU()(x)
        x = self.dropout(x)
        x = self.enc_block_2(x)  # 24,24
        x = nn.SELU()(x)
        x = self.dropout(x)
        x = self.enc_block_3(x)  # 11,11
        x = nn.SELU()(x)
        x = self.dropout(x)
        x = self.enc_block_4(x)  # 5,5
        x = nn.SELU()(x)
        x = self.dropout(x)

        # mu and logvar
        x = self.flatten(x)  # self.hidden_channels *8 *5 * 5
        mu, logvar = self.dense_mu(x), self.dense_logvar(x)

        return mu, logvar


class MarioDecoder(nn.Module):

    def __init__(self, img_channels=3, hidden_channels=32, latent_dim_sub=32, label_dim=9, dropout=0.5):
        super(MarioDecoder, self).__init__()

        self.img_channels = img_channels
        self.hidden_channels = hidden_channels
        self.latent_dim = latent_dim_sub
        self.image_dim = (3, 3)
        self.label_dim = label_dim

        self.dense = nn.Linear(
            in_features=self.latent_dim + self.label_dim,
            out_features=self.hidden_channels * 8 * 5 * 5)

        self.unflatten = UnFlatten()

        self.dec_block_1 = nn.ConvTranspose2d(in_channels=self.hidden_channels * 8,
                                              out_channels=self.hidden_channels * 4,
                                              kernel_size=5,
                                              stride=2,
                                              padding=1)

        self.dec_block_2 = nn.ConvTranspose2d(in_channels=self.hidden_channels * 4,
                                              out_channels=self.hidden_channels * 2,
                                              kernel_size=5,
                                              stride=2,
                                              padding=1,
                                              output_padding=1)

        self.dec_block_3 = nn.ConvTranspose2d(in_channels=self.hidden_channels * 2,
                                              out_channels=self.hidden_channels,
                                              kernel_size=5,
                                              stride=2,
                                              padding=1)

        self.dec_block_4 = nn.ConvTranspose2d(in_channels=self.hidden_channels,
                                              out_channels=self.img_channels,
                                              kernel_size=5,
                                              stride=2,
                                              padding=1,
                                              output_padding=1)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # Unflatten Input
        x = self.dense(x)
        x = self.unflatten(x, self.hidden_channels * 8, dim=(5, 5))

        # MNISTPairsDecoder block 1
        x = self.dec_block_1(x)
        x = nn.SELU()(x)
        x = self.dropout(x)

        # MNISTPairsDecoder block 2
        x = self.dec_block_2(x)
        x = nn.SELU()(x)
        x = self.dropout(x)

        # MNISTPairsDecoder block 3
        x = self.dec_block_3(x)
        x = nn.SELU()(x)
        x = self.dropout(x)

        # MNISTPairsDecoder block 4
        x = self.dec_block_4(x)
        x = torch.nn.Sigmoid()(x)

        return x


class MarioMLP(nn.Module):
    def __init__(self, in_features=20, n_facts=9, hidden_channels=32):
        super(MarioMLP, self).__init__()

        self.n_facts = n_facts
        self.hidden_layer = nn.Linear(in_features=in_features, out_features=hidden_channels)
        self.dense = nn.Linear(in_features=hidden_channels, out_features=n_facts // 2)

    def forward(self, x):
        x = self.hidden_layer(x)
        x = nn.SELU()(x)
        z = self.dense(x)

        return z