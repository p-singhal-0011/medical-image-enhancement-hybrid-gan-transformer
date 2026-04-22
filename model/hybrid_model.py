import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads=4, ff_dim=256):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.view(b, c, -1).permute(2, 0, 1)
        attn_out, _ = self.attn(x, x, x)
        x = self.ln1(x + attn_out)
        ff_out = self.ff(x)
        x = self.ln2(x + ff_out)
        return x.permute(1, 2, 0).view(b, c, h, w)


class GeneratorUNetTransformer(nn.Module):
    def __init__(self):
        super().__init__()

        def down(in_c, out_c, bn=True):
            layers = [nn.Conv2d(in_c, out_c, 4, 2, 1)]
            if bn: layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2))
            return nn.Sequential(*layers)

        def up(in_c, out_c):
            return nn.Sequential(
                nn.ConvTranspose2d(in_c, out_c, 4, 2, 1),
                nn.BatchNorm2d(out_c),
                nn.ReLU()
            )

        self.down1 = down(1, 64, False)
        self.down2 = down(64, 128)
        self.down3 = down(128, 256)
        self.down4 = down(256, 512)

        self.transformer = TransformerBlock(512)

        self.up1 = up(512, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)

        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, 1, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)

        t = self.transformer(d4)

        u1 = self.up1(t)
        u1 = torch.cat([u1, d3], 1)
        u2 = self.up2(u1)
        u2 = torch.cat([u2, d2], 1)
        u3 = self.up3(u2)
        u3 = torch.cat([u3, d1], 1)

        return self.final(u3)
