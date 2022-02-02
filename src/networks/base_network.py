import torch

class BaseNetwork:
    """Neural network object.
    The network object provides an implementation of a function parametrization using a
    neural network.
    This base class provides implementations to common methods to all network objects.
    """

    def __init__(self):
        raise NotImplementedError("This method must be implemented by the subclass")

    @property
    def device(self):
        """torch.device: Determine which device to place the Tensors upon, CPU or GPU."""
        return self.output_layer.weight.device

    @classmethod
    def load(cls, model_path):
        """Load the model from a file."""
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        kwargs = params["kwargs"]
        model = cls(**kwargs)
        model.load_state_dict(params["state_dict"])
        return model

    def copy(self):
        clone = type(self)(**self.kwargs)
        clone.load_state_dict(self.state_dict())
        return clone

    def save(self, path):
        """Save the model to a file."""
        params = {"kwargs": self.kwargs,
                  "state_dict": self.state_dict()
                 }
        torch.save(params, path)

#