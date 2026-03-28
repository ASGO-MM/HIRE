import torch
import torch.nn as nn
import random

# DPO --------------------------------------------
class Actor(nn.Module):
    def __init__(self, i_dim, h_dim=[2048, 1024], o_dim=2, p_dropout=0.5):
        super().__init__()
        cur_dim = i_dim
        layers = []
        for h_d in h_dim:
            layers.append(nn.Linear(cur_dim, h_d))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=p_dropout))
            cur_dim = h_d
        layers.append(nn.Linear(cur_dim, o_dim))
        self.fc = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.fc(x)


class DPOAgent:
    def __init__(self, hidden_dim, latent_dim, action_dim, device, dtype=torch.float32,
                 epsilon_decay=0.995, epsilon_min=0.01, p_dropout=0.5):
        self.actor = Actor(hidden_dim, latent_dim, action_dim, p_dropout=p_dropout).to(device, dtype)
        self.actor.apply(self._kaiming_init_weights)

        self.epsilon = epsilon_decay
        self.gamma = 0.99
        self.epsilon = epsilon_min
        self.dtype = dtype
        self.state_dim=hidden_dim
        self.training=True
        
        self.states=[]
        self.actions=[]

        self.choosen_action = None

    def choose_action(self, state, mask=None):
        if self.training is True:
            bsz = state.shape[0]
            return torch.tensor([random.randint(0, 1) for _ in range(bsz)], dtype=torch.long)
        else:
            if mask is not None:
                state = state[mask]
            state_tensor = state.to(self.dtype)
            q_values = self.actor(state_tensor)
            return q_values.cpu().detach().argmax(-1)

    def train(self):
        self.training=True
        self.actor.train()

    def eval(self):
        self.training=False
        self.actor.eval()

    def load_model(self, model_path):
        checkpoint = torch.load(model_path)
        state_dict = checkpoint['state_dict']
        self.actor.load_state_dict(state_dict)

    def clean_history(self):
        self.states.clear()
        self.actions.clear()

    def _kaiming_init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)