import torch
import einops
import torch.nn as nn
from .unet import UNet_encoder, UNet_decoder
from .model_utils import SoftTimeEmbed, SoftPositionTimeEmbed, attn_mask
from .transformers import TransEncoder, TransDecoder



class BTNTransformerEnc(nn.Module):
    def __init__(self, btn_features, num_layers, num_heads):
        super(BTNTransformerEnc, self).__init__()
        self.encoder_positime = SoftPositionTimeEmbed(btn_features)
        self.transformer_encoder = TransEncoder(btn_features, num_layers, num_heads, dim_feedforward = 512)
    def forward(self, btn):
        masked = True
        b, t, h, w, c = btn.size()
        btn = self.encoder_positime(btn, t, (h, w))  # Position embedding.
        btn = einops.rearrange(btn, 'b t h w c -> b (t h w) c')
        if masked:
            mask = attn_mask(t, (h, w))
            btn = self.transformer_encoder(btn, mask)
        else:
            btn = self.transformer_encoder(btn)
        btn = einops.rearrange(btn, 'b (t h w) c -> b t h w c', t = t, h = h, w = w)
        return btn



class BTNTransformerDec(nn.Module):
    def __init__(self, btn_features, num_layers, num_heads, num_query):
        super(BTNTransformerDec, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.transformer_decoder = TransDecoder(btn_features, num_layers, num_heads, dim_feedforward = 512)
        self.encoder_time = SoftTimeEmbed(btn_features)
        self.motion_embed = nn.Embedding(num_query, btn_features)
        self.num_query = num_query

    def forward(self, btn):
        # b t h w c
        b, t, h, w, c = btn.size()
        btn = einops.rearrange(btn, 'b t h w c -> b (t h w) c')  # Flatten spacetime dimensions. #to: b (t h w) c 
        embed = self.motion_embed(torch.arange(0, self.num_query).expand(b, self.num_query).to(self.device)) #to: b q c 
        embed = embed.unsqueeze(2).repeat(1, 1, t, 1) #to: b q t c 
        embed = self.encoder_time(embed, t) #to: b q t c
        embed = einops.rearrange(embed, 'b q t c -> b (q t) c')
        btn = self.transformer_decoder(embed, btn)
        return btn

class AttnMap(nn.Module):
    def __init__(self, btn_features, num_heads, num_query):
        super(AttnMap, self).__init__()
        self.heads = num_heads
        self.mlp = nn.Sequential(
            nn.Linear(num_heads * num_query, btn_features),
            nn.ReLU(inplace=True))
    def forward(self, btn_dec, btn_enc):
        b, t, h, w, _ = btn_enc.size()
        btn_enc = einops.rearrange(btn_enc, 'b t h w (g c) -> (b g) t h w c', g = self.heads)
        btn_dec = einops.rearrange(btn_dec, 'b (q t) (g c) -> (b g) q t c', t = t, g = self.heads)
        btn_out = btn_dec[:, :, :, None, None, :] * btn_enc[:, None, :, :, :, :]
        btn_out = torch.sum(btn_out, 5) # to: (b g) q t h w 
        btn_out = einops.rearrange(btn_out, '(b g) q t h w -> b (t h w) (q g)', g = self.heads)
        btn_out = self.mlp(btn_out)
        btn_out = einops.rearrange(btn_out, 'b (t h w) c -> b t h w c', t = t, h = h, w = w)
        return btn_out

class OrderingHead(nn.Module):
    def __init__(self, btn_features, num_query):
        super(OrderingHead, self).__init__()
        self.num_query = num_query
        self.mlp = nn.Sequential(
            nn.Linear(btn_features, btn_features),
            nn.ReLU(inplace=True),
            nn.Linear(btn_features, 1))
    def forward(self, btn_dec):
        btn_dec = einops.rearrange(btn_dec, 'b (q t) c -> (b q) t c', q = self.num_query)
        btn_order = torch.max(btn_dec, 1)[0] # to: (b q) c, maxpooling one key frame
        btn_order = self.mlp(btn_order) 
        btn_order = einops.rearrange(btn_order, '(b q) c -> b q c', q = self.num_query)
        return btn_order.squeeze(2)

class OCLR(nn.Module):
    def __init__(self, in_channels, out_channels, unet_features = 32, num_layers = 3, num_heads = 8, num_query = 3):
        super(OCLR, self).__init__()
        btn_features = unet_features * (2 ** 4) 
        self.num_query = num_query
        self.encoder = UNet_encoder(in_channels, unet_features)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.transformer_enc = BTNTransformerEnc(btn_features, num_layers, num_heads)
        self.transformer_dec = BTNTransformerDec(btn_features, num_layers, num_heads, num_query)
        self.heatmap = AttnMap(btn_features, num_heads, num_query)
        self.classhead = OrderingHead(btn_features, num_query)
        self.decoder_am = UNet_decoder(out_channels, unet_features)

    def forward(self, flow):
        # Input flow dim: b t c h w
        # Output segmentation dim: b t c h w
        # Output ordering dim: b t c
        # Notations: batch_size:b; number_of_frames:t; channel_size:c; height:h; width:w; number_of_queries:q
        
        b, t = flow.size()[0:2]
        flow = einops.rearrange(flow, 'b t c h w -> (b t) c h w')
        enc1, enc2, enc3, enc4, btn = self.encoder(flow) # btn dim: (b t) c h w
        btn = einops.rearrange(btn, '(b t) c h w -> b t h w c', b = b)
        btn_enc = self.transformer_enc(btn)
        btn_dec = self.transformer_dec(btn_enc)
        btn_out = self.heatmap(btn_dec, btn_enc)
        btn_out = einops.rearrange(btn_out, 'b t h w c -> (b t) c h w')
        out_am = self.decoder_am(enc1, enc2, enc3, enc4, btn_out)
        out_am =  einops.rearrange(out_am, '(b t) c h w -> b t c h w', b = b)
        out_am = torch.nan_to_num(out_am, nan = 0., posinf = 0., neginf = 0.) # Ignore unstable outputs appearing very occasionally.
        
        out_order = self.classhead(btn_dec)
        b, t, c, _, _ = out_am.size()
        out_order = out_order[:, None].expand(b, t, c) # to: b t c, which copies the same global ordering for all frames.
        out_order = torch.nan_to_num(out_order, nan = 0., posinf = 0., neginf = 0.)
        
        return out_am, out_order
            
