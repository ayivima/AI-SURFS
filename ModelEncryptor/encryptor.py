

import syft
import torch


__author__ = "Victor Mawusi Ayi <ayivima@hotmail.com>"


class ModelEncryptor:

  def __init__(self, model, num_of_shares):

    hook = syft.TorchHook(torch)

    # Function for generating a share
    share = lambda share_id: (
      syft.VirtualWorker(hook, id=share_id).add_worker(syft.local_worker)
    )

    # Generate ids for shares based on number of shares
    share_ids = [
      "".join(["share", str(num + 1)]) 
      for num in range(num_of_shares + 1)
    ]

    # Generate shares based on number of shares specified
    self.shares = list(map(share, share_ids))
    
    # Encrypt model
    self.model = (
      model.fix_precision().share(
        *self.shares[:-1], crypto_provider=self.shares[-1]
      )
    )

  def encrypt_data(self, data):
    """Encrypts data."""
    return (
      data.fix_precision().share(
        *self.shares[:-1], crypto_provider=self.shares[-1]
      )
    )

  def predict(self, data):
    """Encrypts data, and returns the outcome of 
    an encrypted prediction or classification.
    
    """
    e_data = self.encrypt_data(data)

    return self.model(e_data).get().float_precision()


