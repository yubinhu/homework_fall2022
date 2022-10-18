from cs285.critics.sac_critic import SACCritic
from cs285.policies.MLP_policy import MLPPolicy
import torch
import numpy as np
from cs285.infrastructure import sac_utils
from cs285.infrastructure import pytorch_util as ptu
from torch import nn
from torch import optim
from torch import distributions
import itertools

class MLPPolicySAC(MLPPolicy):
    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 discrete=False,
                 learning_rate=3e-4,
                 training=True,
                 log_std_bounds=[-20,2],
                 action_range=[-1,1],
                 init_temperature=1.0,
                 **kwargs
                 ):
        super(MLPPolicySAC, self).__init__(ac_dim, ob_dim, n_layers, size, discrete, learning_rate, training, **kwargs)
        self.log_std_bounds = log_std_bounds
        self.action_range = action_range
        self.init_temperature = init_temperature
        self.learning_rate = learning_rate

        self.log_alpha = torch.tensor(np.log(self.init_temperature)).to(ptu.device)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.learning_rate)

        self.target_entropy = -ac_dim

    @property
    def alpha(self):
        # Formulate temperature term
        return torch.exp(self.log_alpha)

    def get_action(self, obs: np.ndarray, sample=True) -> np.ndarray:
        # TODO: return sample from distribution if sampling
        # if not sampling return the mean of the distribution 
        dist = self.forward(ptu.from_numpy(obs))
        if sample:
            action = ptu.to_numpy(dist.sample())
        else:
            action = ptu.to_numpy(dist.mean.unsqueeze(0))
        return action

    # This function defines the forward pass of the network.
    # You can return anything you want, but you should be able to differentiate
    # through it. For example, you can return a torch.FloatTensor. You can also
    # return more flexible objects, such as a
    # `torch.distributions.Distribution` object. It's up to you!
    def forward(self, observation: torch.FloatTensor):
        # Implement pass through network, computing logprobs and apply correction for Tanh squashing
        # NOTE: adapted from MLP_policy
            # TODO: double check that this makes sense
        if self.discrete:
            logits = self.logits_na(observation)
            action_distribution = distributions.Categorical(logits=logits)
            return action_distribution
        else:
            batch_mean = self.mean_net(observation)
            scale = torch.exp(self.logstd)
            batch_dim = batch_mean.shape[0]
            batch_scale = scale.repeat(batch_dim, 1)
            # TODO: need to check if clip log is needed, also if scale is correct
            action_distribution = sac_utils.SquashedNormal(
                batch_mean,
                batch_scale,
            )

        # HINT: 
        # You will need to clip log values
        # You will need SquashedNormal from sac_utils file 
        return action_distribution

    def update(self, obs, critic:SACCritic):
        # Update actor network and entropy regularizer
        # return losses and alpha value

        obs = ptu.from_numpy(obs)
        action_distribution = self.forward(obs) # shape: (batch, ac_dim)
        
        # actor update
        act_rsample = action_distribution.rsample()
        log_prob = torch.sum(action_distribution.log_prob(act_rsample), dim=-1, keepdim=True)

        Q1, Q2 = critic.forward(obs, act_rsample)
        Q = torch.minimum(Q1, Q2)

        actor_loss = torch.mean(self.alpha.detach() * log_prob - Q )
        # NOTE: using equation 9 in Soft Actor Critic
        # first_term = self.alpha * action_distribution.log_prob(act_rsample)
        # actor_loss1 = torch.mean(first_term - Q1)
        # actor_loss2 = torch.mean(first_term - Q2)
        # actor_loss = actor_loss1 + actor_loss2        
        self.optimizer.zero_grad()
        actor_loss.backward()
        self.optimizer.step()

        # alpha update
        self.log_alpha_optimizer.zero_grad()
        act_sample =  action_distribution.sample()
        log_prob = action_distribution.log_prob(act_sample).detach() # TODO: check if this step is problematic
        
        alpha_loss = (- self.alpha * (log_prob + self.target_entropy).detach()).mean()
        # TODO: double check alpha_loss calculations
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        return actor_loss, alpha_loss, self.alpha