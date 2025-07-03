import torch

class HeadWiseLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, num_heads, bias=True):
        super(HeadWiseLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.weight = torch.nn.Parameter(torch.rand((num_heads, in_features, out_features)))
        self.bias = torch.nn.Parameter(torch.rand((num_heads, out_features))) if bias else None

        # Alternative implementation:
        # self.conv = torch.nn.Conv2d(num_heads*in_features, num_heads*out_features, stride=(1,1), kernel_size=(1,1), groups=num_heads, bias=bias)

    def forward(self, x):
        shape = x.shape
        x = x.reshape((shape[0], shape[1], -1, self.in_features))
        x = torch.einsum("bhsi,hdo->bhso", x, self.weight)
        if self.bias is not None:
            x = x + self.bias.unsqueeze(-2)
        x = x.reshape((shape[0], shape[1], shape[2], shape[3], self.out_features))

        # Alternative implementation:
        # x = x.permute((0, 1, 4, 2, 3)).reshape((shape[0], -1, shape[2], shape[3]))
        # x = self.conv(x)
        return x

class AidedMultiHeadAttention(torch.nn.Module):
    def __init__(self, embed_dim, num_heads:int, aid_depth, dropout:float=0.0, bias:bool=True, mixer_bias:bool=True, batch_first:bool=False, mixer=None):
        super(AidedMultiHeadAttention, self).__init__()
        self.batch_first = batch_first
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = torch.nn.Dropout(dropout)
        if mixer is None:
            self.mixer = torch.nn.Sequential(
                HeadWiseLinear(aid_depth + 1, 1, self.num_heads, bias=mixer_bias),
                torch.nn.ReLU()
            )
        else:
            self.mixer = mixer
        self.linear_out = torch.nn.Linear(embed_dim, embed_dim, bias=bias)
        self.scale = self.head_dim ** -0.5
        self.aid_scale = torch.nn.Parameter(torch.Tensor([0.1]))

    def forward(self, query:torch.Tensor, key:torch.Tensor, value:torch.Tensor, aid:torch.Tensor, attn_mask=None, attn_prev=None):
        if not self.batch_first:
            query, key, value = query.transpose(1, 0), key.transpose(1, 0), value.transpose(1, 0)
            aid = aid.transpose(1, 0)
            attn_mask = attn_mask.transpose(1, 0) if attn_mask is not None else None
            attn_prev = attn_prev.transpose(1, 0) if attn_prev is not None else None

        aid = aid.unsqueeze(1).repeat_interleave(self.num_heads, dim=1)

        query = query.reshape(query.size(0), self.num_heads, query.size(1), self.head_dim)
        key = key.reshape(key.size(0), self.num_heads, key.size(1), self.head_dim)
        value = value.reshape(value.size(0), self.num_heads, value.size(1), self.head_dim)

        attn = torch.matmul(query, key.transpose(2, 3))

        attn = attn * self.scale
        attn = attn + self.mixer(torch.cat([attn.unsqueeze(-1), aid], dim=-1)).squeeze(-1) * self.aid_scale

        attn = self.dropout(attn)

        if attn_mask is not None:
            if attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1)
            attn = attn.masked_fill(attn_mask == 0, -1e9)

        pre_softmax_attn = attn
        attn = attn.softmax(dim=-1)
        attn = attn + attn_prev if attn_prev is not None else attn
        output = torch.matmul(attn, value)
        output = output.transpose(1, 2).reshape(output.size(0), output.size(2), self.embed_dim)
        output = self.linear_out(output)

        if not self.batch_first:
            output = output.transpose(1, 0)
            pre_softmax_attn = pre_softmax_attn.transpose(1, 0)
            attn = attn.transpose(1, 0)
        hiddens = {
            "attn" : attn,
            "pre_softmax_attn" : pre_softmax_attn
        }
        return output, hiddens

class AidedAttentionLayer(torch.nn.Module):
    def __init__(self, embed_dim, num_heads, aid_depth, dropout:float=0.0, attn_bias:bool=True, mixer_bias:bool=True, batch_first:bool=False, ff_mult:int=4, mixer=None):
        super(AidedAttentionLayer, self).__init__()
        self.attn = AidedMultiHeadAttention(embed_dim, num_heads, aid_depth, dropout, attn_bias, mixer_bias, batch_first=batch_first, mixer=mixer)
        self.norm1 = torch.nn.LayerNorm(embed_dim)
        self.norm2 = torch.nn.LayerNorm(embed_dim)
        self.to_q = torch.nn.Linear(embed_dim, embed_dim)
        self.to_k = torch.nn.Linear(embed_dim, embed_dim)
        self.to_v = torch.nn.Linear(embed_dim, embed_dim)
        self.feed_forward = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, embed_dim*ff_mult),
            torch.nn.GELU(),
            torch.nn.Linear(embed_dim*ff_mult, embed_dim)
        )

    def forward(self, x, y, aid, attn_mask=None, x_mask=None, y_mask=None, attn_prev=None):
        # what about pre-norm?
        if x_mask is not None:
            if y_mask is None:
                y_mask = x_mask
            input_mask = (x_mask.float().unsqueeze(-1)).bmm(y_mask.float().unsqueeze(-1).transpose(-1,-2)).long().unsqueeze(1)
            if attn_mask is None:
                attn_mask = input_mask
            else:
                attn_mask = attn_mask & input_mask
        output1, hiddens = self.attn(self.to_q(x), self.to_k(y), self.to_v(y), aid, attn_mask=attn_mask, attn_prev=attn_prev)
        output1 = self.norm1(x + output1)
        output2 = self.feed_forward(output1)
        output2 = self.norm2(output1 + output2)
        return output2, hiddens

class AidedTransformer(torch.nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, aid_depth, dropout=0.0, batch_first=False, mixer=None, residual=False):
        super(AidedTransformer, self).__init__()
        self.layers = torch.nn.ModuleList([AidedAttentionLayer(embed_dim, num_heads, aid_depth, dropout, batch_first=batch_first, mixer=mixer) for _ in range(num_layers)])
        self.residual=residual

    def forward(self, x, y, aid, attn_mask=None, x_mask=None, y_mask=None, return_hiddens=False):
        attn_hiddens = []
        for layer in self.layers:
            if self.residual and len(attn_hiddens) > 0:
                attn_prev = attn_hiddens[-1]['attn']
            else:
                attn_prev = None
            x, hiddens = layer(x, y, aid=aid, attn_mask=attn_mask, x_mask=x_mask, y_mask=y_mask, attn_prev=attn_prev)
            attn_hiddens.append(hiddens)
        if return_hiddens:
            return x, attn_hiddens
        else:
            return x

if __name__ == "__main__":
    # Test the AidedTransformer module
    model = AidedTransformer(embed_dim=512, num_heads=8, num_layers=6, aid_depth=2)
    x = torch.randn(10, 32, 512)
    y = torch.randn(5, 32, 512)
    distance = torch.randn(10, 32, 5)
    connectivity = torch.randn(10, 32, 5)
    output = model(x, y, torch.cat((distance.unsqueeze(-1), connectivity.unsqueeze(-1)), dim=-1))
    print(output.shape)
