import torch
import torch.nn as nn


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class PyBulletEncoder(nn.Module):
    """ PyBullet Encoder from Original Paper's Keras based Implementation.

        Args:
            init_num_filters (int): initial number of filters from encoder image channels
            lrelu_slope (float): positive number indicating LeakyReLU negative slope
            embedding_dim (int): embedding dimensionality
    """
    def __init__(self, init_num_filters=64, lrelu_slope=0.2, embedding_dim=64):
        super(PyBulletEncoder, self).__init__()

        self.init_num_filters_ = init_num_filters
        self.lrelu_slope_ = lrelu_slope
        self.embedding_dim_ = embedding_dim

        self.features = nn.Sequential(
            # 3x48x48
            nn.Conv2d(3, self.init_num_filters_, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(self.lrelu_slope_, inplace=True),

            # Cx24x24
            nn.Conv2d(self.init_num_filters_, self.init_num_filters_ * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.init_num_filters_ * 2),
            nn.LeakyReLU(self.lrelu_slope_, inplace=True),

            # 2Cx12x12
            nn.Conv2d(self.init_num_filters_ * 2, self.init_num_filters_ * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.init_num_filters_ * 4),
            nn.LeakyReLU(self.lrelu_slope_, inplace=True),

            # 4Cx6x6
            nn.Conv2d(self.init_num_filters_ * 4, self.init_num_filters_ * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.init_num_filters_ * 8),
            nn.LeakyReLU(self.lrelu_slope_, inplace=True),

            # 8Cx3x3
            # nn.BatchNorm2d(self.init_num_filters_ * 4),
            # nn.LeakyReLU(self.lrelu_slope_, inplace=True),
            nn.Conv2d(self.init_num_filters_ * 8, self.embedding_dim_, kernel_size=4, stride=2, padding=1, bias=True),
            # Ex3x3
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, self.embedding_dim_)
        return x


class PyBulletDecoder(nn.Module):
    """ PyBullet Decoder from Original Paper's Keras based Implementation.

        Args:
            init_num_filters (int): initial number of filters from encoder image channels
            lrelu_slope (float): positive number indicating LeakyReLU negative slope
            embedding_dim (int): embedding dimensionality
    """
    def __init__(self, init_num_filters=64, embedding_dim=64):
        super(PyBulletDecoder, self).__init__()

        self.init_num_filters_ = init_num_filters
        self.embedding_dim_ = embedding_dim

        self.features = nn.Sequential(

            nn.ConvTranspose2d(self.embedding_dim_, self.init_num_filters_ * 8, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.init_num_filters_ * 8),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(self.init_num_filters_ * 8, self.init_num_filters_ * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.init_num_filters_ * 4),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(self.init_num_filters_ * 4, self.init_num_filters_ * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.init_num_filters_ * 2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(self.init_num_filters_ * 2, self.init_num_filters_, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.init_num_filters_),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(self.init_num_filters_, 3, kernel_size=4, stride=2, padding=1, bias=True),
        )

    def forward(self, z):
        z = z.view(-1, self.embedding_dim_, 1, 1)
        z = self.features(z)
        return torch.sigmoid(z)


class PyBulletAutoencoder(nn.Module):
    """ PyBullet Autoencoder from Original Paper's Keras based Implementation.

        Args:
            init_num_filters (int): initial number of filters from encoder image channels
            lrelu_slope (float): positive number indicating LeakyReLU negative slope
            embedding_dim (int): embedding dimensionality
    """
    def __init__(self, init_num_filters=64, lrelu_slope=0.2, embedding_dim=64):
        super(PyBulletAutoencoder, self).__init__()

        self.init_num_filters_ = init_num_filters
        self.lrelu_slope_ = lrelu_slope
        self.embedding_dim_ = embedding_dim

        self.encoder = PyBulletEncoder(init_num_filters, lrelu_slope, embedding_dim)
        self.decoder = PyBulletDecoder(init_num_filters, embedding_dim)

        self.encoder.apply(weights_init)
        self.decoder.apply(weights_init)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z
