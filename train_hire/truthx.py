import torch
from torch import nn
import torch.nn.functional as F
from abc import abstractmethod
from torch import tensor as Tensor
from typing import List, Any


class BaseVAE(nn.Module):

    def __init__(self) -> None:
        super(BaseVAE, self).__init__()

    def encode(self, input: Tensor) -> List[Tensor]:
        raise NotImplementedError

    def decode(self, input: Tensor) -> Any:
        raise NotImplementedError

    def sample(self, batch_size: int, current_device: int, **kwargs) -> Tensor:
        raise NotImplementedError

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
        pass

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> Tensor:
        pass


class MLPAE(BaseVAE):
    def __init__(
        self,
        in_channels: int,
        semantic_latent_dim: int,
        truthful_latent_dim: int,
        semantic_hidden_dims: List = None,
        truthful_hidden_dims: List = None,
        decoder_hidden_dims: List = None,
        num_layers: int = 16,
        **kwargs
    ) -> None:
        super(MLPAE, self).__init__()

        self.semantic_latent_dim = semantic_latent_dim

        if semantic_hidden_dims is None:
            semantic_hidden_dims = []

        # Build Semantic Encoder
        semantic_encoder_modules = []
        flat_size = in_channels
        for h_dim in semantic_hidden_dims:
            semantic_encoder_modules.append(
                nn.Sequential(
                    nn.Linear(flat_size, h_dim), nn.LayerNorm(h_dim), nn.LeakyReLU()
                )
            )
            flat_size = h_dim
        semantic_encoder_modules.append(
            nn.Sequential(
                nn.Linear(flat_size, semantic_latent_dim),
                nn.LayerNorm(semantic_latent_dim),
                nn.LeakyReLU(),
            )
        )

        self.semantic_encoder = nn.Sequential(*semantic_encoder_modules)

        if truthful_hidden_dims is None:
            truthful_hidden_dims = []

        # Build Truthful Encoder
        truthful_encoder_modules = []
        flat_size = in_channels
        for h_dim in truthful_hidden_dims:
            truthful_encoder_modules.append(
                nn.Sequential(
                    (
                        nn.Linear(flat_size, h_dim)
                        if flat_size != h_dim
                        else nn.Identity()
                    ),
                    nn.LayerNorm(h_dim),
                    nn.LeakyReLU(),
                )
            )
            flat_size = h_dim
        truthful_encoder_modules.append(
            nn.Sequential(
                (
                    nn.Linear(flat_size, truthful_latent_dim)
                    if flat_size != truthful_latent_dim
                    else nn.Identity()
                ),
                nn.LayerNorm(truthful_latent_dim),
                nn.LeakyReLU(),
            )
        )

        self.truthful_encoder = nn.Sequential(*truthful_encoder_modules)

        # Cross-Attention Module
        self.num_heads = 1
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=semantic_latent_dim, num_heads=self.num_heads
        )

        self.proj = None
        if semantic_latent_dim != truthful_latent_dim:
            self.proj = nn.Linear(truthful_latent_dim, semantic_latent_dim, bias=False)

        # Build Decoder
        decoder_modules = []
        if len(decoder_hidden_dims) > 0:
            flat_size = semantic_latent_dim
            for h_dim in decoder_hidden_dims:
                decoder_modules.append(
                    nn.Sequential(
                        nn.Linear(flat_size, h_dim), nn.LayerNorm(h_dim), nn.LeakyReLU()
                    )
                )
                flat_size = h_dim

            flat_size = decoder_hidden_dims[-1]
            self.decoder = nn.Sequential(*decoder_modules)
        else:
            self.decoder_input = None

            self.decoder = None
            flat_size = semantic_latent_dim
        self.final_layer = nn.Sequential(nn.Linear(flat_size, in_channels))

        # self.layer_embedding = nn.Embedding(num_layers, in_channels)


    def encode_semantic(self, input: Tensor) -> List[Tensor]:
        semantic_latent_rep = self.semantic_encoder(input)
        return semantic_latent_rep

    def encode_truthful(self, input: Tensor) -> List[Tensor]:
        truthful_latent_rep = self.truthful_encoder(input)
        truthful_latent_rep = F.normalize(truthful_latent_rep, p=2, dim=-1)

        return truthful_latent_rep

    def attention(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        if self.proj is not None and query.size(-1) != key.size(-1):
            key = self.proj(key)
            value = self.proj(value)
        query = query.unsqueeze(0)
        key = key.unsqueeze(0)
        value = value.unsqueeze(0)

        output, attention_weights = self.cross_attention(query, key, value)

        return output[0]

    def decode(self, z: Tensor) -> Tensor:
        result = z
        if self.decoder is not None:
            result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def forward(
        self, input: Tensor, truthful_latent_rep=None, layer_index=None, **kwargs
    ) -> List[Tensor]:
    
        semantic_latent_rep = self.encode_semantic(input)
        if truthful_latent_rep is None:
            truthful_latent_rep = self.encode_truthful(input)
        truthful_latent_rep = truthful_latent_rep.reshape(
            -1, truthful_latent_rep.size(-1)
        )
        z = semantic_latent_rep + self.attention(
            semantic_latent_rep,
            truthful_latent_rep.contiguous(),
            truthful_latent_rep.contiguous(),
        )
        output = self.decode(z)

        return [output, input, semantic_latent_rep, truthful_latent_rep]
    
    def recon(
            self, semantic, truthful, **kwargs
    ) -> Tensor:
        truthful = truthful.reshape(
            -1, truthful.size(-1)
        )
        z = semantic + self.attention(
            semantic,
            truthful.contiguous(),
            truthful.contiguous(),
        )
        output = self.decode(z)

        return output

    def forward_decoder(self, input, semantic_latent_rep, truthful_latent_rep):
        z = semantic_latent_rep + self.attention(
            semantic_latent_rep, truthful_latent_rep, truthful_latent_rep
        )
        output = self.decode(z)
        return [output, input, semantic_latent_rep, truthful_latent_rep]

    def get_semantic_latent_rep(self, input: Tensor, **kwargs) -> List[Tensor]:
        semantic_latent_rep = self.encode_semantic(input)
        return semantic_latent_rep

    def get_truthful_latent_rep(self, input: Tensor, **kwargs) -> List[Tensor]:
        truthful_latent_rep = self.encode_truthful(input)
        return truthful_latent_rep

    def loss_function(self, *args, **kwargs) -> dict:
        recons = args[0]
        input = args[1]
        recons_loss = F.mse_loss(recons, input)

        loss = recons_loss
        return {"loss": loss, "Reconstruction_Loss": recons_loss.detach()}            


class TruthX:
    def __init__(self, model_path=None, hidden_size=768, latent_dim=None, num_layers=16, train_layer=None, edit_strength=1.0, dtype=None, device=None):
        self.latent_dim=latent_dim
        if dtype is None:
            self.dtype = torch.float32
        else:
            self.dtype = dtype
        if model_path is not None:
            checkpoint = torch.load(model_path, map_location=device)
        if latent_dim is None:
            semantic_latent_dim = 1024  # Set default value
            truthful_latent_dim = 1024  # Set default value
            semantic_hidden_dims = [2048, 1024]  # Set default architecture
            truthful_hidden_dims = [2048, 1024]  # Set default architecture
            decoder_hidden_dims = [1024, 2048]  # Set default architecture
        else:
            truthful_hidden_dims = latent_dim
            semantic_hidden_dims = latent_dim
            decoder_hidden_dims = latent_dim[::-1]
            semantic_latent_dim = latent_dim[-1]
            truthful_latent_dim = latent_dim[-1]

        # Check if a valid model_path is provided
        if model_path:
            ae_model = MLPAE(
                in_channels=hidden_size,
                semantic_latent_dim=semantic_latent_dim,
                truthful_latent_dim=truthful_latent_dim,
                semantic_hidden_dims=semantic_hidden_dims,
                truthful_hidden_dims=truthful_hidden_dims,
                decoder_hidden_dims=decoder_hidden_dims,
                num_layers = num_layers,
            ).to(device, dtype=self.dtype)

            ae_model.load_state_dict(checkpoint["state_dict"])
            print(f"load ae_model state_dict successfully!")
            if 'pos_center' in checkpoint.keys():
                ae_model.pos_center = checkpoint.get("pos_center", torch.zeros(truthful_latent_dim))
                ae_model.neg_center = checkpoint.get("neg_center", torch.zeros(truthful_latent_dim))

                for layer in ae_model.pos_center.keys():
                    ae_model.pos_center[layer] = ae_model.pos_center[layer].to(device) 
                for layer in ae_model.neg_center.keys():
                    ae_model.neg_center[layer] = ae_model.neg_center[layer].to(device)

                print("pos and neg center load!")
            self.train_layer = checkpoint["train_layer"]
            self.early_stop_layer = int(max(self.train_layer) / 2)
        else:
            ae_model = MLPAE(
                in_channels=hidden_size,
                semantic_latent_dim=semantic_latent_dim,
                truthful_latent_dim=truthful_latent_dim,
                semantic_hidden_dims=semantic_hidden_dims,
                truthful_hidden_dims=truthful_hidden_dims,
                decoder_hidden_dims=decoder_hidden_dims,
                num_layers=num_layers
            ).to(device, dtype=self.dtype)

            if train_layer is None:
                raise ValueError(f"train_layer must be list, but got {type(train_layer)}")
            self.train_layer = train_layer

            self.early_stop_layer = int(max(train_layer) / 2)
            ae_model.pos_center = {rank_id: torch.zeros(semantic_latent_dim, device=device) for i, rank_id in enumerate(self.train_layer)}
            ae_model.neg_center = {rank_id: torch.zeros(semantic_latent_dim, device=device) for i, rank_id in enumerate(self.train_layer)}


        ae_model.train()
        self.ae_model = ae_model
        self.edit_strength = edit_strength
        self.cur_layer_id = 0
        self.num_edit_token = None
        self.get_activation = False
        self.training=False

        self.target_dtype = self.ae_model.semantic_encoder[0][0].weight.dtype
        self.deltas = {}
        # for layer_id in self.train_layer:
        #     p = self.ae_model.pos_center[layer_id]
        #     n = self.ae_model.neg_center[layer_id]
        #     # 预先计算并转为目标 dtype
        #     self.deltas[layer_id] = (p - n).view(1, 1, -1).to(device)

    def train(self):
        self.ae_model.train()
        self.training=True

    def eval(self):
        self.ae_model.eval()
        self.training=False

    def check_edit(self):
        layer_id = int(self.cur_layer_id.split(".")[0])
        if self.cur_layer_id.endswith("attn"):
            layer_id = 2 * layer_id
        else:
            layer_id = 2 * layer_id + 1
        if layer_id not in self.train_layer:
            return False
        else:
            return True

    @torch.inference_mode()
    def edit(self, X):
        layer_id = int(self.cur_layer_id.split(".")[0])
        if self.cur_layer_id.endswith("attn"):
            layer_id = 2 * layer_id
        else:
            layer_id = 2 * layer_id + 1

        if layer_id not in self.train_layer:
            return X

        bsz, s_len, d = X.size()
        x = (
            X.contiguous()
            .view(-1, d)
            .type_as(self.ae_model.semantic_encoder[0][0].weight)
        )
        x_truthful = self.ae_model.get_truthful_latent_rep(
            X.type_as(self.ae_model.semantic_encoder[0][0].weight)
        )

        pos_center = self.ae_model.pos_center[layer_id].unsqueeze(0)
        neg_center = self.ae_model.neg_center[layer_id].unsqueeze(0)
        delta = (pos_center - neg_center).unsqueeze(0)            

        recon_x_pos = (
            self.ae_model(
                x,
                truthful_latent_rep=F.normalize(
                    x_truthful + delta, p=2, dim=-1
                ).type_as(x),
            )[0]
            .contiguous()
            .view(bsz, s_len, d)
        )
        recon_x_neg = (
            self.ae_model(
                x,
                truthful_latent_rep=F.normalize(
                    x_truthful - delta, p=2, dim=-1
                ).type_as(x),
            )[0]
            .contiguous()
            .view(bsz, s_len, d)
        )

        Delta = recon_x_pos - recon_x_neg
        Delta = Delta.contiguous().to(X.dtype)
        Delta = F.normalize(Delta, p=2, dim=-1).type_as(X) * torch.norm(
            X, p=2, dim=-1
        ).unsqueeze(2)

        mask = torch.ones((bsz, s_len), device=Delta.device)
        
        mask[:, :-1] = 0
        if self.num_edit_token is not None:
            mask[:, -self.num_edit_token:] = 1
        else:
            mask[:, -1:] = 1

        new_X = X + (Delta.type_as(X)) * self.edit_strength * mask.unsqueeze(2).type_as(X)

        if torch.isinf(new_X).any() or torch.isnan(new_X).any():
            print("error")

        return new_X