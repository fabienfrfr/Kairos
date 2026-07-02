import torch
from torch import nn

class UnifiedVQVAE(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(UnifiedVQVAE, self).__init__()

        # Dimension of embedding vectors
        self.embedding_dim = embedding_dim
        # Number of unique tokens in the codebook
        self.num_embeddings = num_embeddings

        # Encoder network (shared across modalities) --> futur trick, multiple input (3D = video/audio, 2D = image/wav, 1D = text/signal)
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, embedding_dim, kernel_size=1, stride=1)
        )

        # Decoder network (shared across modalities) --> futur trick, multiple output (3D = video/audio, 2D = image/wav, 1D = text/signal)
        self.decoder = nn.Sequential(
            nn.Conv1d(embedding_dim, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 1, kernel_size=3, stride=1, padding=1)
        )

        # Codebook for vector quantization
        self.codebook = nn.Embedding(num_embeddings, embedding_dim)
        # Initialize codebook weights uniformly
        self.codebook.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def encode(self, x):
        # Encode input to latent space
        z = self.encoder(x)
        return z

    def decode(self, z):
        # Decode latent representation back to input space
        return self.decoder(z)

    def quantize(self, z):
        # Flatten the latent representation
        z_flattened = z.view(-1, self.embedding_dim)

        # Calculate distances between latent vectors and codebook entries
        distances = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
                    torch.sum(self.codebook.weight ** 2, dim=1) - \
                    2 * torch.matmul(z_flattened, self.codebook.weight.t())

        # Find the closest codebook entries
        encoding_indices = torch.argmin(distances, dim=1)
        quantized = self.codebook(encoding_indices).view(z.shape)
        return quantized, encoding_indices

    def forward(self, x):
        # Encode input
        z = self.encode(x)
        # Quantize latent representation
        quantized, indices = self.quantize(z)
        # Reconstruct input
        x_recon = self.decode(quantized)
        return x_recon, quantized, indices