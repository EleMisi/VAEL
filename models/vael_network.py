from torch import nn


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, input, hidden_channels, dim):
        return input.view(input.size(0), hidden_channels * 4, dim[0], dim[1])


class Encoder(nn.Module):

    def __init__(self, img_channels=1, hidden_channels=32, latent_dim=8, dropout=0.5):
        super(Encoder, self).__init__()

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
        # Encoder block 1
        x = self.enc_block_1(x)
        x = nn.ReLU()(x)
        x = self.dropout(x)

        # Encoder block 2
        x = self.enc_block_2(x)
        x = nn.ReLU()(x)
        x = self.dropout(x)

        # Encoder block 3
        x = self.enc_block_3(x)
        x = nn.ReLU()(x)

        # mu and logvar
        x = self.flatten(x)  # batch_size, dim1, dim2, dim3 -> batch_size, dim1*dim2*dim3
        mu, logvar = self.dense_mu(x), self.dense_logvar(x)

        return mu, logvar


class Decoder(nn.Module):

    def __init__(self, img_channels=1, hidden_channels=32, latent_dim=32, label_dim=9, dropout=0.5,
                 **params):
        super(Decoder, self).__init__()

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
        x = self.unflatten(x, self.hidden_channels, self.unflatten_dim)

        # Decoder block 1
        x = self.dec_block_1(x)
        x = nn.ReLU()(x)
        x = self.dropout(x)

        # Decoder block 2
        x = self.dec_block_2(x)
        x = nn.ReLU()(x)
        x = self.dropout(x)

        # Decoder block 3
        x = self.dec_block_3(x)

        return x


class MLP(nn.Module):
    def __init__(self, in_features=20, n_digits=10, hidden_channels=20):
        super(MLP, self).__init__()

        self.n_digits = n_digits
        self.hidden_layer = nn.Linear(in_features=in_features, out_features=hidden_channels)
        self.dense_z1 = nn.Linear(in_features=hidden_channels, out_features=n_digits)
        self.dense_z2 = nn.Linear(in_features=hidden_channels, out_features=n_digits)

    def forward(self, x):
        x = self.hidden_layer(x)
        x = nn.ReLU()(x)
        z1 = self.dense_z1(x)
        z2 = self.dense_z2(x)

        return z1, z2
