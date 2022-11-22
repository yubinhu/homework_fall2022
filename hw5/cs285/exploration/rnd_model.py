from cs285.infrastructure import pytorch_util as ptu
from .base_exploration_model import BaseExplorationModel
import torch.optim as optim
from torch import nn
import torch

def init_method_1(model):
    model.weight.data.uniform_()
    model.bias.data.uniform_()

def init_method_2(model):
    model.weight.data.normal_()
    model.bias.data.normal_()


class RNDModel(nn.Module, BaseExplorationModel):
    def __init__(self, hparams, optimizer_spec, **kwargs):
        super().__init__(**kwargs)
        self.ob_dim = hparams['ob_dim']
        self.output_size = hparams['rnd_output_size']
        self.n_layers = hparams['rnd_n_layers']
        self.size = hparams['rnd_size']
        self.optimizer_spec = optimizer_spec

        # <DONE>: Create two neural networks:
        # 1) f, the random function we are trying to learn
        # 2) f_hat, the function we are using to learn f
        self.f = ptu.build_mlp(self.ob_dim, self.output_size, self.n_layers, self.size, 
                               init_method=init_method_1).to(ptu.device)
        self.f_hat = ptu.build_mlp(self.ob_dim, self.output_size, self.n_layers, self.size, 
                                   init_method=init_method_2).to(ptu.device)
        self.optimizer = self.optimizer_spec.constructor(
            self.f_hat.parameters(),
            **self.optimizer_spec.optim_kwargs
        )

    def forward(self, ob_no:torch.Tensor):
        # Get the prediction error for ob_no
        # HINT: Remember to detach the output of self.f!
        goal = self.f.forward(ob_no).detach()
        pred = self.f_hat.forward(ob_no)
        err = torch.norm(goal-pred, dim=1) # TODO: check this norm and err shape
        assert err.shape[0] == ob_no.shape[0]
        return err
        

    def forward_np(self, ob_no):
        ob_no = ptu.from_numpy(ob_no)
        error = self(ob_no)
        return ptu.to_numpy(error)

    def update(self, ob_no):
        # <DONE>: Update f_hat using ob_no
        # Hint: Take the mean prediction error across the batch
        err = self.forward(ptu.from_numpy(ob_no))
        loss = err.mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return ptu.to_numpy(loss)
