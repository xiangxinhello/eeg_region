from einops.layers.torch import Rearrange, Reduce
from dc_ldm.Fre_Time_FuseTransformer import *
# Transformer Parameters
d_model = 128  # Embedding Size


# 对decoder的输入来屏蔽未来信息的影响，这里主要构造一个矩阵
def get_attn_subsequence_mask(seq):
    '''
    seq: [batch_size, tgt_len]
    '''
    # 创建一个三维矩阵
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    # triu产生上三角矩阵，k=1对角线的位置上移一个，k=0表示正常的上三角，k=-1表示对角线位置下移1个对角线
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)  # Upper triangular matrix
    # byte转换成只有0和1，上三角全是1，下三角全是0
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    # subsequence_mask = torch.from_numpy(subsequence_mask).byte().cuda()
    # subsequence_mask = subsequence_mask.data.eq(0)  # 0-True,1-False
    return subsequence_mask  # [batch_size, tgt_len, tgt_len]


# Attention过程
# 通过 Q 和 K 计算出 scores，然后将 scores 和 V 相乘，得到每个单词的 context vector
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        '''
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)  # scores : [batch_size, n_heads, len_q, len_k]
        # mask为True的位置全部填充为-无穷，无效区域不参与softmax计算
        if attn_mask is not None:
            scores.masked_fill_(attn_mask, -1e9)  # Fills elements of self tensor with value where mask is True.

        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)  # [batch_size, n_heads, len_q, d_v]
        return context, attn


# 多头Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self, input_Q, input_K, input_V, attn_mask):
        '''
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        residual, batch_size = input_Q, input_Q.size(0)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1,
                                                                    2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1,
                                                  1)  # attn_mask : [batch_size, n_heads, seq_len, seq_len]

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)
        context = context.transpose(1, 2).reshape(batch_size, -1,
                                                  n_heads * d_v)  # context: [batch_size, len_q, n_heads * d_v]
        # 转换d_model维度
        output = self.fc(context)  # [batch_size, len_q, d_model]
        d_model = output.shape[2]
        # 残差连接+LN
        return nn.LayerNorm(d_model).cuda()(output + residual), attn


# 前馈神经网络
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.GELU()
        )
        # Swish激活函数
        # Swish激活函数
        # self.swish = My_Swish()
        self.batchNorm = nn.BatchNorm1d(d_ff)

        self.fc2 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(d_ff, d_model, bias=False),
        )

    def forward(self, inputs):
        '''
        inputs: [batch_size, seq_len, d_model]
        '''
        residual = inputs

        input_fc1 = self.fc(inputs)

        # Swish激活函数
        # input_fc1_sw = self.swish(input_fc1)

        # (b, t, c) - > (b, c, t)
        # input_fc1_sw = input_fc1_sw.permute(0, 2, 1)
        input_fc1_sw = input_fc1.permute(0, 2, 1)
        input_bn = self.batchNorm(input_fc1_sw)
        # (b, t, c) - > (b, c, t)
        input_bn = input_bn.permute(0, 2, 1)

        output = self.fc2(input_bn)
        d_model = output.shape[2]
        # imageNet_images = self.fc(inputs)
        return nn.LayerNorm(d_model).cuda()(output + residual)  # [batch_size, seq_len, d_model]
#         # return nn.BatchNorm1d(d_model).cuda()(imageNet_images + residual)  # [batch_size, seq_len, d_model]

# class PoswiseFeedForwardNet(nn.Module):
#     def __init__(self):
#         super(PoswiseFeedForwardNet, self).__init__()
#         self.fc = nn.Sequential(
#             nn.Linear(d_model, d_ff, bias=False),
#             nn.ReLU(),
#             # nn.Dropout(0.1),
#             nn.Linear(d_ff, d_model, bias=False)
#         )
#
#     def forward(self, inputs):
#         '''
#         inputs: [batch_size, seq_len, d_model]
#         '''
#         residual = inputs
#         imageNet_images = self.fc(inputs)
#         return nn.LayerNorm(d_model).cuda()(imageNet_images + residual)  # [batch_size, seq_len, d_model]
#         # return nn.BatchNorm1d(d_model).cuda()(imageNet_images + residual)  # [batch_size, seq_len, d_model]


def fourier_transform(x):
    # return torch.fft.fft2(x, signal_ndim=(-1, -2)).real
    # return torch.fft(torch.fft(x, signal_ndim=3), signal_ndim=2).real
    output = x.cuda().data.cpu().numpy()
    # imageNet_images = np.fft.fft2(imageNet_images, axes=(-1, -2)).real
    output = np.fft.fft(np.fft.fft(output, axis=-1), axis=-2).real
    output = torch.Tensor(output)
    return output.cuda()


class EncoderLayer(nn.Module):
    def __init__(self, d_model):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(d_model)
        self.pos_ffn = PoswiseFeedForwardNet(d_model)

    def forward(self, enc_inputs, enc_self_attn_mask):
        '''
        enc_inputs: [batch_size, src_len, d_model]
        enc_self_attn_mask: [batch_size, src_len, src_len]
        '''
        # enc_outputs: [batch_size, src_len, d_model], attn: [batch_size, n_heads, src_len, src_len]
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs,
                                               enc_self_attn_mask)  # enc_inputs to same Q,K,V
        # FFT Features
        # enc_outputs += fourier_transform(enc_outputs)
        enc_outputs = self.pos_ffn(enc_outputs)  # enc_outputs: [batch_size, src_len, d_model]
        return enc_outputs, attn



class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.src_emb = nn.Linear(128, time_d_model ,bias=False)
        # self.pat_emb = PatchEmbedding(d_model)

        self.pos_emb = nn.Parameter(torch.randn(1, 444, time_d_model))
        self.cls_token = nn.Parameter(torch.randn(1, 4, time_d_model))
        # self.cls_token = nn.Parameter(torch.randn(1, 1, 192))

        self.dropout = nn.Dropout(0.2)
        self.layers = nn.ModuleList([EncoderLayer(d_model=time_d_model) for _ in range(n_layers)])

    def forward(self, enc_inputs):
        '''
        enc_inputs: [batch_size, src_len, d_model]
        '''
        # [b, t, c] - > [b, 1, c, t]
        # enc_inputs = enc_inputs.unsqueeze(1).transpose(2,3)
        # # Patchembedding:[b, 1, c, t] - > [b, t, c]
        # enc_outputs = self.pat_emb(enc_inputs)

        # Time Transformer do this
        enc_outputs = self.src_emb(enc_inputs)
        # enc_outputs = enc_inputs
        # cls_token
        b, n, _ = enc_outputs.shape
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        enc_outputs = torch.cat((cls_tokens, enc_outputs), dim=1)

        # Position embedding
        b, n, _ = enc_outputs.shape
        enc_outputs += self.pos_emb[:,:n]
        enc_outputs = self.dropout(enc_outputs)

        # encoder中没有pad不需要mask
        # enc_self_attn_mask = get_attn_pad_mask(enc_outputs, enc_outputs)  # [batch_size, src_len, src_len]
        enc_self_attn_mask = None
        enc_self_attns = []

        for layer in self.layers:
            # enc_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns

class Channel_Encoder(nn.Module):
    def __init__(self):
        super(Channel_Encoder, self).__init__()
        self.src_emb = nn.Linear(440, d_model ,bias=False)
        self.pos_emb = nn.Parameter(torch.randn(1, 132, d_model))
        self.cls_token = nn.Parameter(torch.randn(1, 4, d_model))
        # self.cls_token = nn.Parameter(torch.randn(1, 1, 256))

        self.dropout = nn.Dropout(0.2)
        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model) for _ in range(n_layers)])

    def forward(self, enc_inputs):
        '''
        enc_inputs: [batch_size, src_len, d_model]
        '''
        # [b, t, c] - > [b, 1, c, t]
        # enc_inputs = enc_inputs.unsqueeze(1).transpose(2,3)
        # # Patchembedding:[b, 1, c, t] - > [b, t, c]
        # enc_outputs = self.pat_emb(enc_inputs)

        # Channel Transformer do this
        enc_outputs = self.src_emb(enc_inputs)
        # cls_token
        b, n, _ = enc_outputs.shape
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        enc_outputs = torch.cat((cls_tokens, enc_outputs), dim=1)
        # Position embedding
        enc_outputs += self.pos_emb[:,:(n+4)]
        enc_outputs = self.dropout(enc_outputs)

        # encoder中没有pad不需要mask
        # enc_self_attn_mask = get_attn_pad_mask(enc_outputs, enc_outputs)  # [batch_size, src_len, src_len]
        enc_self_attn_mask = None
        enc_self_attns = []

        for layer in self.layers:
            # enc_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns

class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size, n_classes):
        super().__init__()
        self.clshead = nn.Sequential(
            # Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size,emb_size * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(emb_size * 2, emb_size),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(emb_size, n_classes)
        )

    def forward(self, x):
        #x=torch.flatten(x,dim=0)
        out = self.clshead(x)
        # return x, out
        return out

class Frequency_Encoder(nn.Module):
    def __init__(self):
        super(Frequency_Encoder, self).__init__()
        self.src_emb = nn.Linear(1, d_model, bias=False)
        self.pos_emb = nn.Parameter(torch.randn(1, 668, d_model))
        self.cls_token = nn.Parameter(torch.randn(1, 4, d_model))
        self.dropout = nn.Dropout(0.2)
        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model) for _ in range(n_layers)])
    def forward(self, enc_inputs):
        enc_inputs = enc_inputs.to(torch.float32)
        # enc_inputs:[b, f, d]
        enc_inputs = self.src_emb(enc_inputs)
        # add class_token
        b, n, _ = enc_inputs.shape
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        enc_inputs = torch.cat((cls_tokens, enc_inputs), dim=1)
        # Position embedding
        enc_inputs += self.pos_emb[:, :(n+4)]
        enc_pos_inputs = self.dropout(enc_inputs)
        # mask
        enc_self_attn_mask = None
        enc_self_attns = []
        # encoder
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_pos_inputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)

        return enc_outputs, enc_self_attns

class time_attention(nn.Module):
    def __init__(self, sequence_num=128, inter=6, time_size = 440):
        super(time_attention, self).__init__()
        self.sequence_num = sequence_num
        self.inter = inter
        self.extract_sequence = int(self.sequence_num / self.inter)  # You could choose to do that for less computation

        self.query = nn.Sequential(
            nn.Linear(time_size, time_size),
            nn.LayerNorm(time_size),  # also may introduce improvement to a certain extent
            nn.Dropout(0.2)
        )
        self.key = nn.Sequential(
            nn.Linear(time_size, time_size),
            # nn.LeakyReLU(),
            nn.LayerNorm(time_size),
            nn.Dropout(0.2)
        )

        # self.value = self.key
        self.projection = nn.Sequential(
            nn.Linear(time_size, time_size),
            # nn.LeakyReLU(),
            nn.LayerNorm(time_size),
            nn.Dropout(0.2),
        )

        self.drop_out = nn.Dropout(0.2)
        self.pooling = nn.AvgPool2d(kernel_size=(1, self.inter), stride=(1, self.inter))
        self.drop_out_last = nn.Dropout(0.2)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        # [b t c] -> [b 1 t c]
        x = x.unsqueeze(1)
        # 实质上是[b 1 t c] -> [b 1 c t] 字符cs是原来channel_attention的没有改
        temp = rearrange(x, 'b o c s->b o s c')
        # add layernorm
        temp = nn.LayerNorm(temp.shape[3]).cuda()(temp)

        temp_query = rearrange(self.query(temp), 'b o s c -> b o c s')
        temp_key = rearrange(self.key(temp), 'b o s c -> b o c s')

        channel_query = self.pooling(temp_query)
        channel_key = self.pooling(temp_key)

        scaling = self.extract_sequence ** (1 / 2)

        channel_atten = torch.einsum('b o c s, b o m s -> b o c m', channel_query, channel_key) / scaling

        channel_atten_score = torch.softmax(channel_atten, dim=-1)
        channel_atten_score = self.drop_out(channel_atten_score)

        out = torch.einsum('b o c s, b o c m -> b o c s', x, channel_atten_score)
        '''
        projections after or before multiplying with attention score are almost the same.
        '''
        out = rearrange(out, 'b o c s -> b o s c')
        out = self.projection(out)
        out = rearrange(out, 'b o s c -> b o c s')
        # return out

        # Add: dropout + residual block
        out = self.drop_out_last(out)
        res = out + x
        # res:[b 1 t c]
        return res.squeeze(1)

class FNetEncoderLayer(nn.Module):
    def __init__(self, d_model, expansion_factor, dropout):
        super().__init__()
        self.ff = PoswiseFeedForwardNet(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        x = fourier_transform(x)
        x = self.norm1(x + residual)
        residual = x
        x = self.ff(x)
        out = self.norm2(x + residual)
        return out


class FNet(nn.Module):
    def __init__(
        self, d_model=256, expansion_factor=2, dropout=0.5, num_layers=6,
    ):
        super().__init__()
        # encoder_layer
        encoder_layer = FNetEncoderLayer(d_model, expansion_factor, dropout)
        # embedding
        self.embedding = nn.Linear(22, d_model)
        # 加token
        self.cls_token = nn.Parameter(torch.randn(1, 4, d_model))
        # LocalEncoding
        self.pos_emb = nn.Parameter(torch.randn(1, 1129, 32))
        # self.encoder = nn.Embedding(intoken, hidden)
        # cls_token_fc
        self.cls_token_fc = nn.Linear(4, 1)
        # classify
        self.mlp_head = nn.Sequential(
            nn.Linear(32, 64),
            nn.Dropout(0.1),
            # nn.GELU(),
            nn.ReLU(),
            nn.Linear(64, 4),
        )
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # 交换t和c的位置
        # [b, c, t] -> [b, t, c]
        x = x.permute(0, 2, 1)
        b, t, c = x.shape
        # embedding
        output_embedding = self.embedding(x)
        # # 加token
        cls_token = self.cls_token.repeat(b, 1, 1)
        # [b, t, c] -> [b, t+1, c]
        output_pos_embedding = torch.cat((cls_token, output_embedding), dim=1)
        # output_pos_embedding += self.pos_emb[:, :(t + 4)]
        # output_pos_embedding = self.dropout(output_pos_embedding)
        encoder_layer = output_pos_embedding
        # encoder_layer
        for layer in self.layers:
            encoder_layer = layer(encoder_layer)
        # 取token进行分类[b, t+1, c] -> [b, token, c]
        output = encoder_layer[:, :4, :]
        # [b, token, c] -> [b, c, token] -> [b, c, 1]-> [b, c]
        output = self.cls_token_fc(output.permute(0, 2, 1)).squeeze(2)
        # depthwise_conv
        # imageNet_images = self.depthwise_conv(encoder_layer).squeeze(1)
        # classify
        output = self.mlp_head(output)
        return output


class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        # self.channel_attention = channel_attention(channel_size=96).cuda()
        # self.encoder = Encoder().cuda()
        # self.channel_encoder = Channel_Encoder().cuda()
        # self.fre_encoder = Frequency_Encoder().cuda()
        self.Fre_Time_Fuse_encoder = FT_Fuse_FuseEncoder()
        # self.to_latent = nn.Identity()
        # RELU激活函数
        # self.mlp_head = nn.Sequential(
        #     nn.Linear(d_model,2048, bias=False),
        #     nn.ReLU(),
        #     nn.Dropout(0.6),
        #     nn.Linear(2048, 1024, bias=False),
        #     nn.Linear(1024, 512, bias=False),
        #     nn.ReLU(),
        #     nn.Dropout(0.6),
        #     nn.Linear(512, 192, bias=False),
        #     nn.Linear(192, 26, bias=False)
        #
        # )
        #
        # # Swish激活函数
        # self.swish = My_Swish()
        # self.dropout = nn.Dropout(0.6)
        # self.mlp_head_1 = nn.Sequential(
        #     nn.Linear(d_model,2048, bias=False),
        # )
        #
        # self.mlp_head_2 = nn.Sequential(
        #     nn.Linear(2048, 1024, bias=False),
        #     nn.Linear(1024, 512, bias=False),
        # )
        # self.mlp_head_3 = nn.Sequential(
        #     nn.Linear(512, 192, bias=False),
        #     nn.Linear(192, 26, bias=False)
        # )

        self.classification = ClassificationHead(128, 40)
        torch.manual_seed(2022)
        self.label = torch.randint(0, 10, (4,), device="cuda:0")
        self.class_embedding = nn.Embedding(40, 128)

    def forward(self, fre_enc_inputs, time_enc_inputs):
        '''
        enc_inputs: [batch_size, src_len, d_model]
        dec_inputs: [batch_size, tgt_len, d_model]
        '''

        # enc_outputs: [batch_size, src_len, d_model], enc_self_attns: [n_layers, batch_size, n_heads, src_len, src_len]

        # split channel attention
        # channel attention [b, t, c] - > [b, 1, c, t]
        # enc_inputs = enc_inputs.unsqueeze(1).transpose(2,3)
        # enc_inputs_firstarray = enc_inputs[:,:,:96,:]
        # enc_inputs_secondarray = enc_inputs[:,:,96:,:]
        # enc_inputs_firstarray_channel_attention = self.channel_attention(enc_inputs_firstarray)
        # enc_inputs_secondarray_channel_attention = self.channel_attention(enc_inputs_secondarray)
        # enc_inputs_channel_attention = torch.cat((enc_inputs_firstarray_channel_attention, enc_inputs_secondarray_channel_attention), dim=2)

        # whole channel attention
        # channel attention [b, t, c] - > [b, 1, c, t]
        # enc_inputs = enc_inputs.unsqueeze(1).transpose(2, 3)
        # enc_inputs_channel_attention = self.channel_attention(enc_inputs)

        #  [b, 1, c, t] - > [b, t, c]
        # enc_inputs_channel_attention = enc_inputs_channel_attention.squeeze(1).transpose(1,2)
        # enc_outputs, enc_self_attns = self.encoder(enc_inputs_channel_attention)

        # Time transformer [no cls token, no src embedding] enc_outputs = [8, 200, 192]
        # enc_inputs_time = time_enc_inputs
        # enc_time_outputs, time_self_attns = self.encoder(enc_inputs_time)

        # channel transformer
        # enc_channel_inputs = time_enc_inputs + enc_time_outputs
        # enc_channel_inputs = rearrange(enc_channel_inputs, 'b t c->b c t')
        # enc_outputs_channel, enc_self_attns = self.channel_encoder(enc_channel_inputs)


        # frequency transformer
        # enc_fre_inputs = fre_enc_inputs
        # enc_fre_outputs, fre_self_attns = self.fre_encoder(enc_fre_inputs)

        # frequency time fuse transformer
        enc_FT_outputs, FT_self_attens = self.Fre_Time_Fuse_encoder(fre_enc_inputs, time_enc_inputs)

        # cls_token cat
        # cls_output_FT = enc_FT_outputs
        cls = Reduce('b n e -> b e', reduction='mean')(enc_FT_outputs)
        # torch.manual_seed(2022)
        # labels = torch.randint(0, 10, (16,), device="cuda")
        class_embedding = self.class_embedding(self.label)
        class_embedding = class_embedding + cls
        # cls_output_FT = enc_FT_outputs.view(enc_FT_outputs.size(0),-1)
        # cls_output_fre = enc_fre_outputs[:, :4]
        # cls_output_time = enc_time_outputs[:, :4]
        # cls_output_channel = enc_outputs_channel[:,:4]
        # cls_output = enc_outputs[:, :1]
        # cls_output = torch.cat((cls_output, cls_output_channel), dim=1)
        # cls_output = cls_output + cls_output_channel
        # cls_output = enc_outputs
        # feature concat
        # cls_output = torch.cat((cls_output_fre, cls_output_time), dim=1)

        # res_output = self.classification(class_embedding)
        return class_embedding
