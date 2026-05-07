import torch
import torch.nn as nn
import torch.nn.functional as F


torch.set_num_threads(4)

class MaskGuidedFusionBlock(nn.Module):

    def __init__(self, channels=320, num_res=3):
        super().__init__()
        self.C = channels
        self.num_res = num_res

        self.mr1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.mr2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.mr3 = nn.Conv2d(128, channels, kernel_size=3, padding=1)
        self.mr_act = nn.Sigmoid()

        self.f_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.softsign = nn.Softsign()
        self.ig1 = nn.Conv2d(channels, channels, kernel_size=1)
        self.ig2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

        mid = channels // 2
        self.res3x3 = nn.ModuleList([
            nn.Conv2d(channels, mid, kernel_size=3, padding=1) for _ in range(num_res)
        ])
        self.res1x1 = nn.ModuleList([
            nn.Conv2d(mid, channels, kernel_size=1) for _ in range(num_res)
        ])

        self.out_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, m, f):

        l = F.relu(self.mr1(m), inplace=True)
        l = F.relu(self.mr2(l), inplace=True)
        l = self.mr3(l)
        l = self.mr_act(l) 

        x = F.relu(self.ig1(f), inplace=True)
        x = F.relu(self.ig2(x), inplace=True)

        for k in range(self.num_res):
            r = F.relu(self.res3x3[k](x), inplace=True)
            r = self.res1x1[k](r)
            x = x + r
        x = self.softsign(torch.tanh(x))
        x = x * l
        x = x + f
        return x


class Mask2TextModel(nn.Module):
    def __init__(self, pretrained_model_name, hidden_dim=256):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(pretrained_model_name, subfolder="text_encoder")

        text_hidden = self.text_encoder.config.hidden_size
        vocab_size = self.tokenizer.vocab_size

        self.mask_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, hidden_dim),
            nn.ReLU()
        )

        self.fusion = nn.Linear(text_hidden + hidden_dim, text_hidden)

        self.mlm_head = nn.Linear(text_hidden, vocab_size)

    def forward(self, sentences, binary_masks, mask_labels):

        device = next(self.parameters()).device

        # 1. tokenize
        encodings = self.tokenizer(
            sentences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        )
        input_ids = encodings.input_ids.to(device)
        attention_mask = encodings.attention_mask.to(device)

        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        text_hidden = text_outputs.last_hidden_state  

        mask_feat = binary_masks.unsqueeze(1).float().to(device) 
        mask_feat = self.mask_encoder(mask_feat)  

        mask_feat_exp = mask_feat.unsqueeze(1).expand(-1, text_hidden.size(1), -1)

        fused = torch.cat([text_hidden, mask_feat_exp], dim=-1)
        fused_hidden = self.fusion(fused)  


        logits = self.mlm_head(fused_hidden)  

        mask_labels = mask_labels.to(device)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            mask_labels.view(-1),
            ignore_index=-100
        )

        return loss, logits



if __name__ == "__main__":
    pretrained_model = ""
    model = Mask2TextModel(pretrained_model).cuda()

    sentences = [
        "a car is parked under a tree",
        "a man is walking with a dog"
    ]


    binary_masks = torch.randint(0, 2, (2, 64, 64)).cuda()

    enc = model.tokenizer(sentences, return_tensors="pt", padding=True, truncation=True, max_length=77)
    input_ids = enc.input_ids.clone()
    labels = input_ids.clone()

    labels[:] = -100
    labels[0, 2] = input_ids[0, 2]  # mask "car"
    input_ids[0, 2] = model.tokenizer.mask_token_id
    labels[0, 5] = input_ids[0, 5]  # mask "tree"
    input_ids[0, 5] = model.tokenizer.mask_token_id
    labels[1, 6] = input_ids[1, 6]  # mask "dog"
    input_ids[1, 6] = model.tokenizer.mask_token_id

    loss, logits = model(sentences, binary_masks, labels)
    print("Loss:", loss.item())
    print("Logits:", logits.shape)  # (B,L,V)
















#----------test-------------
# b = 2
# m = torch.randint(0, 2, (b, 16, 32))        # (b,16,32)
# f = torch.randn(b, 320, 16, 32)             # (b,320,16,32)

# block = MaskGuidedFusionBlock(channels=320, num_res=3)
# out = block(m, f)
# print(out.shape)  # -> (2, 320, 16, 32)


# class AlignFeedBack(nn.Module):

#     # _supports_gradient_checkpointing = True

#     # @register_to_config 
#     def __init__(
#             self,
#             in_channels: int = 4,
#             H=128,
#             W=256,
#             N=192,
#             M=320
#     ):
#         super().__init__()
#         self.ca1 = CrossAttention(input_size=(H // 16, W // 16), num_filters=N,
#                                   dim=256, num_patches=4, heads=8, dropout=0.1)
#         self.ca2 = CrossAttention(input_size=(H // 16, W // 16), num_filters=N,
#                                   dim=256, num_patches=4, heads=8, dropout=0.1)
#         self.ca3 = CrossAttention(input_size=(H // 16, W // 16), num_filters=M,
#                                   dim=256, num_patches=4, heads=8, dropout=0.1)
        
#         self.down_x1 = nn.Sequential(
#             conv(in_channels, N),
#             ResidualBottleneckBlock(N),
#             ResidualBottleneckBlock(N),
#             ResidualBottleneckBlock(N),
#             )
#         self.down_x2 = nn.Sequential(    
#             conv1x1(2*N, N),
#             ResidualBottleneckBlock(N),
#             ResidualBottleneckBlock(N),
#             ResidualBottleneckBlock(N),
#             )
#         self.down_x3 = nn.Sequential(  
#             conv1x1(2*N, N),
#             ResidualBottleneckBlock(N),
#             ResidualBottleneckBlock(N),
#             ResidualBottleneckBlock(N),
#             conv1x1(N, M),
#             AttentionBlock(M)
#         )

#         self.down_y1 = nn.Sequential(
#             conv(in_channels, N),
#             ResidualBottleneckBlock(N),
#             ResidualBottleneckBlock(N),
#             ResidualBottleneckBlock(N),
#             )
#         self.down_y2 = nn.Sequential(    
#             conv1x1(N, N),
#             ResidualBottleneckBlock(N),
#             ResidualBottleneckBlock(N),
#             ResidualBottleneckBlock(N)
#             )
#         self.down_y3 = nn.Sequential(  
#             conv1x1(N, N),
#             ResidualBottleneckBlock(N),
#             ResidualBottleneckBlock(N),
#             ResidualBottleneckBlock(N),
#             conv1x1(N, M),
#             AttentionBlock(M)
#         )
#         self.up_x1 = nn.Sequential(  
#             nn.ConvTranspose2d(2*M, M, 5, stride=2, padding=2, output_padding=1),
#             ResidualBottleneckBlock(M),
#             ResidualBottleneckBlock(M),
#             ResidualBottleneckBlock(M),
#             AttentionBlock(M),
#         )   
#         self.up_x2 = nn.Sequential(  
#             conv1x1(M, N),
#             ResidualBottleneckBlock(N),
#             ResidualBottleneckBlock(N),
#             ResidualBottleneckBlock(N),
#             AttentionBlock(N),
#         )               
#         self.up_x3 = nn.Sequential(  
#             conv1x1(N, N),
#             ResidualBottleneckBlock(N),
#             ResidualBottleneckBlock(N),
#             ResidualBottleneckBlock(N),
#             AttentionBlock(N),
#             conv1x1(N, 4),
#         )   
        
#     def forward(self, x: Tensor, y: Tensor) -> Tensor:

#         x = self.down_x1(x)
#         y = self.down_y1(y)
#         x = self.ca1(x, y)
        
#         x = self.down_x2(x)
#         y = self.down_y2(y)
#         x = self.ca2(x, y)
        
#         x = self.down_x3(x)
#         y = self.down_y3(y)
#         x = self.ca3(x, y)

#         x = self.up_x1(x)
#         x = self.up_x2(x)
#         x = self.up_x3(x)

#         return x
    
# class CSIfusion(nn.Module):
#     # @register_to_config
#     def __init__(self, dim=320, num_heads=8, depth=2):
#         super().__init__()
#         self.dim = dim

#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=dim, nhead=num_heads, batch_first=True
#         )
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

#         # 可训练的位置编码（相对位置 bias 会更复杂，这里用简单可训练 embedding）
#         self.pos_embed = nn.Parameter(torch.randn(1, 10000, dim))  # 支持最多10000长度序列

#         self.proj_out = nn.Linear(dim, dim)
#         self.token_proj = None 

#     def forward(self, z_hat_cor, z_hat_target, common_mask):
#         """
#         Args:
#             z_hat_cor: (B,C,H,W)
#             z_hat_target: (B,C,H1,W1)
#             common_mask: (B,H,W) 0/1
#         Return:
#             fused_target: (B,C,H1,W1)
#         """
#         B, C, H, W = z_hat_cor.shape
#         _, _, H1, W1 = z_hat_target.shape

#         # flatten (B,HW,C)
#         cor_flat = z_hat_cor.flatten(2).transpose(1, 2)         # (B,HW,C)
#         tgt_flat = z_hat_target.flatten(2).transpose(1, 2)      # (B,H1W1,C)

#         # mask 公共部分
#         mask_flat = common_mask.flatten(1).unsqueeze(-1)        # (B,HW,1)
#         cor_common = cor_flat * mask_flat                       # (B,HW,C)

#         # 拼接序列 (B,L,C)
#         fused = torch.cat([cor_common, tgt_flat], dim=1)

#         # 加位置编码 (自动截断)
#         pos = self.pos_embed[:, :fused.shape[1], :]
#         fused = fused + pos

#         # transformer 融合
#         fused_out = self.transformer(fused)  # (B,L,C)
#         if self.token_proj is None:
#             self.token_proj = nn.Linear(fused_out.shape[1], H1*W1).to(fused_out.device)

#         fused_out = fused_out.transpose(1, 2)      # (B,C,L)
#         fused_out = self.token_proj(fused_out)     # (B,C,target_len)
#         fused_out = fused_out.transpose(1, 2)      # (B,target_len,C)

#         # channel-wise 映射
#         fused_out = self.proj_out(fused_out)       # (B,target_len,C)

#         # reshape 回 target 空间
#         fused_target = fused_out.transpose(1, 2).reshape(B, C, H1, W1)

#         return fused_target


# class CommonPrior(nn.Module):
    
#     """
#     超先验函数，输入 z_hat_cor，输出每个位置是否为公共类别的概率
#     """
#     # @register_to_config
#     def __init__(self, in_channels, hidden_dim=512):
#         super().__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(hidden_dim, 1, 3, padding=1)   # 输出 1 通道
#         )

#     def forward(self, z_hat_cor):
#         """
#         z_hat_cor: (B, C, H, W)   # VQ-VAE 的 quantized latent
#         return: probs (B, H, W) ∈ [0,1]
#         """
#         logits = self.conv(z_hat_cor)           # (B, 1, H, W)
#         probs = torch.sigmoid(logits).squeeze(1)  # (B, H, W)
#         return probs


