import torch

from torchfm.layer import FactorizationMachine, FeaturesEmbedding, FeaturesLinear


class FactorizationMachineModel(torch.nn.Module):
    """
    A pytorch implementation of Factorization Machine.

    Reference:
        S Rendle, Factorization Machines, 2010.
    """

    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.linear = FeaturesLinear(field_dims)
        self.fm = FactorizationMachine(reduce_sum=True)

    def forward(self, x, v):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        :param v: Float tensor of size ``(batch_size, num_fields)``
        """
        x = self.linear(x, v) + self.fm(self.embedding(x, v))
        return torch.sigmoid(x.squeeze(1))
