import numpy as np
from einops import rearrange, repeat
import torch.nn as nn
import torch
from dc_ldm.BAM import *
from dc_ldm.transformer_configure import opt
import math
# Transformer Parameters
channel_size = 128
time_d_model = 128  # Time Embedding Size
d_model = 128  # Embedding Size
d_ff = 2048  # FeedForward dimension (62-256-62线性提取的过程)
d_k = d_v = 128 # dimension of K(=Q), V
n_layers = 1  # number of Encoder of Decoder Layer
n_heads = 32


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
            nn.Dropout(0.4),
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
        # output = self.fc(inputs)
        return nn.LayerNorm(d_model).cuda()(output + residual)  # [batch_size, seq_len, d_model]
#         # return nn.BatchNorm1d(d_model).cuda()(output + residual)  # [batch_size, seq_len, d_model]

class Flayers(nn.Module):
    def __init__(self, channels, reduction = 2):
        super(Flayers, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels,channels//reduction,bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels//reduction, channels, bias=False),
            nn.Sigmoid()

        )
    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool (x) . view (b,c)
        y = self.fc(y).view(b,c,1)
        return x * y.expand_as(x)




class FT_Fuse_EncoderLayer(nn.Module):
    def __init__(self, d_model):
        super(FT_Fuse_EncoderLayer, self).__init__()
        self.Fre_self_attn = MultiHeadAttention(d_model)
        self.Time_self_attn = MultiHeadAttention(d_model)
        self.pos_ffn = PoswiseFeedForwardNet(d_model)
        self.fusion=Flayers(2)

    def forward(self, Fre_pos_inputs,Time_pos_inputs, enc_self_attn_mask):
        '''
        enc_inputs: [batch_size, src_len, d_model]
        enc_self_attn_mask: [batch_size, src_len, src_len]
        '''
        # enc_outputs: [batch_size, src_len, d_model], attn: [batch_size, n_heads, src_len, src_len]
        Fre_outputs, attn = self.Fre_self_attn(Fre_pos_inputs, Fre_pos_inputs, Fre_pos_inputs,
                                               enc_self_attn_mask)  # enc_inputs to same Q,K,V
        Time_outputs, attn = self.Time_self_attn(Time_pos_inputs, Time_pos_inputs, Time_pos_inputs,
                                               enc_self_attn_mask)  #
        Fre_outputs=Fre_outputs[:,:1]
        Time_outputs=Time_outputs[:,:1]
        #enc_outputs=torch.matmul(Fre_outputs,Time_outputs)
        enc_outputs=torch.cat((Fre_outputs,Time_outputs),dim=1)
        #enc_outputs=self.fusion(enc_outputs)
        #enc_outputs = torch.cat((Fre_outputs, Time_outputs), dim=1)
        # enc_outputs = torch.cat((Time_outputs, Fre_outputs), dim=1)
        # FFT Features
        # enc_outputs += fourier_transform(enc_outputs)
        enc_outputs = self.pos_ffn(enc_outputs)  # enc_outputs: [batch_size, src_len, d_model]
        enc_outputs=self.fusion(enc_outputs)
        return enc_outputs, attn


class FT_BAM_Fuse_FuseEncoder(nn.Module):
    def __init__(self):
        super(FT_BAM_Fuse_FuseEncoder, self).__init__()
        self.Fre_src_emb = nn.Linear(1, d_model, bias=False)
        self.Fre_pos_emb = nn.Parameter(torch.randn(1, 664, d_model))
        self.Fre_cls_token = nn.Parameter(torch.randn(1, 4, d_model))
        self.Time_cls_token = nn.Parameter(torch.randn(1, 4, d_model))
        self.Time_src_emb = nn.Linear(128, time_d_model, bias=False)
        self.Time_pos_emb = nn.Parameter(torch.randn(1, 444, time_d_model))
        self.dropout = nn.Dropout(0.2)
        self.layers = nn.ModuleList([FT_Fuse_EncoderLayer(d_model=d_model) for _ in range(n_layers)])
        self.my_bam = BAM(d_model)
    def forward(self, Fre_inputs, Time_inputs):
        Fre_inputs = Fre_inputs.to(torch.float32)
        Time_inputs = Time_inputs.to(torch.float32)
        # Fre
        Fre_inputs = self.Fre_src_emb(Fre_inputs)
        # add class_token
        b, n, _ = Fre_inputs.shape
        # cls_tokens = repeat(self.Fre_cls_token, '() n e -> b n e', b=b)
        # Fre_inputs = torch.cat((cls_tokens, Fre_inputs), dim=1)
        # Position embedding
        # Fre_inputs += self.Fre_pos_emb[:, :(n + 4)]
        # Fre_inputs += self.Fre_pos_emb[:, :n]
        Fre_inputs_bam = rearrange(Fre_inputs, 'b f c -> b c f')
        Fre_inputs_bam = Fre_inputs_bam.unsqueeze(2)
        BAM_Fre_outputs = self.my_bam(Fre_inputs_bam)
        BAM_Fre_outputs = BAM_Fre_outputs.squeeze(2)
        BAM_Fre_outputs = rearrange(BAM_Fre_outputs, 'b c f -> b f c')
        Fre_inputs = Fre_inputs + BAM_Fre_outputs
        Fre_pos_inputs = self.dropout(Fre_inputs)
        # Time
        b, n, _ = Time_inputs.shape
        Time_inputs = self.Time_src_emb(Time_inputs)
        cls_tokens = repeat(self.Time_cls_token, '() n e -> b n e', b=b)
        Time_inputs = torch.cat((cls_tokens, Time_inputs), dim=1)
        # Time_inputs += self.Time_pos_emb[:, :(n+4)]
        Time_inputs_bam = rearrange(Time_inputs, 'b t c -> b c t')
        Time_inputs_bam = Time_inputs_bam.unsqueeze(2)
        Time_inputs_bam = self.my_bam(Time_inputs_bam)
        Time_inputs_bam = Time_inputs_bam.squeeze(2)
        Time_inputs_bam = rearrange(Time_inputs_bam, 'b c t -> b t c')
        Time_inputs = Time_inputs + Time_inputs_bam
        Time_pos_inputs = self.dropout(Time_inputs)


        enc_self_attn_mask = None
        enc_self_attns = []

        # enc_outputs = torch.cat((Fre_pos_inputs, Time_pos_inputs), dim=1)

        for layer in self.layers:
            # enc_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            enc_outputs, enc_self_attn = layer(Fre_pos_inputs, Time_pos_inputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns

class channel_attention(nn.Module):
    def __init__(self, sequence_num=400, channel_size = 128, inter=10):
        super(channel_attention, self).__init__()
        self.sequence_num = sequence_num
        self.inter = inter
        self.extract_sequence = int(self.sequence_num / self.inter)  # You could choose to do that for less computation

        self.query = nn.Sequential(
            nn.Linear(channel_size, channel_size),
            nn.LayerNorm(channel_size),  # also may introduce improvement to a certain extent
            nn.Dropout(0.3)
        )
        self.key = nn.Sequential(
            nn.Linear(channel_size, channel_size),
            # nn.LeakyReLU(),
            nn.LayerNorm(channel_size),
            nn.Dropout(0.3)
        )

        # self.value = self.key
        self.projection = nn.Sequential(
            nn.Linear(channel_size, channel_size),
            # nn.LeakyReLU(),
            nn.LayerNorm(channel_size),
            nn.Dropout(0.3),
        )

        self.drop_out = nn.Dropout(0)
        self.pooling = nn.AvgPool2d(kernel_size=(1, self.inter), stride=(1, self.inter))
        self.drop_out_last = nn.Dropout(0.5)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        temp = rearrange(x, 'b o c s->b o s c')
        temp = temp.to('cuda:0')
        # add layerNorm
        temp = nn.LayerNorm(channel_size).cuda()(temp)

        temp_query = rearrange(self.query(temp), 'b o s c -> b o c s')
        temp_key = rearrange(self.key(temp), 'b o s c -> b o c s')

        channel_query = self.pooling(temp_query)
        channel_key = self.pooling(temp_key)

        scaling = self.extract_sequence ** (1 / 2)

        channel_atten = torch.einsum('b o c s, b o m s -> b o c m', channel_query, channel_key) / scaling

        channel_atten_score = F.softmax(channel_atten, dim=-1)
        # ====================save channel_atten_score
        # if(opt.current_epoch == 1 or opt.current_epoch % 50 == 0):
            #np.save("/home/stu/lcs/weizhan-eeg/egg/code/LCS_TEarea_FT_Transformer/beforF_channelAttention_map/attention_map_time_" + '%d' % opt.current_epoch + ".npy",
            #         channel_atten_score[0, :, :, :].detach().cpu().numpy())
        # ====================end save channel_atten_score
        channel_atten_score = self.drop_out(channel_atten_score).to('cuda:0')
        x = x.to('cuda:0')
        out = torch.einsum('b o c s, b o c m -> b o c s', x, channel_atten_score)
        '''
        projections after or before multiplying with attention score are almost the same.
        '''
        out = rearrange(out, 'b o c s -> b o s c')
        out = self.projection(out)
        out = rearrange(out, 'b o s c -> b o c s')

        # Add: dropout + residual block
        out = self.drop_out_last(out)
        res = out + x
        return res
class lear_tokens(nn.Module):
    def __init__(self,d_model,n_tokens):
        super(lear_tokens, self).__init__()
        self.tokens_l = nn.Sequential(nn.Linear(d_model,n_tokens,bias=False),
                                   nn.LayerNorm(n_tokens),
                                   nn.Linear(n_tokens, n_tokens, bias=False),
                                   nn.GELU(),
                                   nn.Linear(n_tokens, n_tokens, bias=False),
                                   nn.GELU(),
                                   nn.Linear(n_tokens, n_tokens, bias=False),

            )


    def forward(self,inputs):
        """Applies learnable tokenization to the 2D inputs.

    Args:
      inputs: Inputs of shape `[bs, h, w, c]` or `[bs, hw, c]`.

    Returns:
      Output of shape `[bs, n_token, c]`.
    """
        # inputs=inputs.permute(0,2,1)
        #b,c,t=inputs.size()
        # if inputs.ndim == 3:
        #     n, hw, c = inputs.shape
        #     # h = int(math.sqrt(hw))
        #     inputs = torch.reshape(inputs, [n, 2, -1, c])

            # if h * h != hw:
            #     raise ValueError('Only square inputs supported.')

        feature_shape = inputs.shape

        selected = inputs
        selected = selected.to('cuda:0')
        selected=self.tokens_l(selected)

        # selected = self.layer_n(selected)
        # selected = self.fc_down(selected)
        #
        # for _ in range(3):
        #     # selected = nn.Conv(
        #     #     self.num_tokens,
        #     #     kernel_size=(3, 3),
        #     #     strides=(1, 1),
        #     #     padding='SAME',
        #     #     use_bias=False)(selected)  # Shape: [bs, h, w, n_token].
        #     selected=nn.Linear(n_token,n_token)(selected)
        #     selected = nn.GELU(selected.cuda())
        #
        # # selected = nn.Conv(
        # #     self.num_tokens,
        # #     kernel_size=(3, 3),
        # #     strides=(1, 1),
        # #     padding='SAME',
        # #     use_bias=False)(selected)  # Shape: [bs, h, w, n_token].
        # selected=nn.Linear(n_token,n_token)

        # selected = torch.reshape(
        #     selected, [feature_shape[0], feature_shape[1] * feature_shape[2],
        #                ])  # Shape: [bs, h*w, n_token].
        selected = selected.permute(0,2,1) # Shape: [bs, n_token, h*w].
        selected =nn.Softmax()(selected)
        # selected=selected.unsqueeze(3)# Shape: [bs, n_token, h*w, 1].

        feat = inputs
        # feat = torch.reshape(
        #     feat, [feature_shape[0], feature_shape[1] * feature_shape[2], -1
        #            ])[:, None, ...]  # Shape: [bs, 1, h*w, c].
        #
        # if self.use_sum_pooling:
        #     inputs = torch.sum(feat * selected, axis=2)
        # else:
        #     inputs = torch.mean(feat * selected, axis=2)
        feat = feat.to('cuda:0')
        feat = torch.einsum('...si,...id->...sd', selected, feat)

        return feat


class FT_Fuse_FuseEncoder(nn.Module):
    def __init__(self):
        super(FT_Fuse_FuseEncoder, self).__init__()
        self.Fre_src_emb = nn.Linear(1, d_model, bias=False)
        self.Fre_pos_emb = nn.Parameter(torch.randn(1, 664, d_model))
        self.Fre_cls_token = nn.Parameter(torch.randn(1, 4, d_model))
        self.Time_cls_token = nn.Parameter(torch.randn(1, 4, d_model))
        # self.Fre_cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        # self.Time_cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.Time_src_emb = nn.Linear(128, time_d_model, bias=False)
        self.Time_pos_emb = nn.Parameter(torch.randn(1, 444, time_d_model))
        #self.Time_pos_emb = nn.Parameter(torch.randn(1, 147, time_d_model))#444
        self.dropout = nn.Dropout(0.2)
        self.layers = nn.ModuleList([FT_Fuse_EncoderLayer(d_model=d_model) for _ in range(n_layers)])
        self.my_bam = BAM(d_model)
        self.F_channel_atten = channel_attention(5, inter=2)
        self.T_channel_atten = channel_attention(400)
        self.conv=nn.Sequential(nn.Conv2d(1,3,kernel_size=(3,1),stride=(3,1)),
                                nn.BatchNorm2d(3),
                                nn.ReLU(),
                                nn.Conv2d(3, 1, kernel_size=1, bias=False),
                                nn.BatchNorm2d(1),
                                nn.ReLU()
                                )
        self.lear_t=lear_tokens(128,128//3)
        self.lear_f = lear_tokens(128, 128 // 3)
    def forward(self, Fre_inputs, Time_inputs):
        Fre_inputs = Fre_inputs.to(torch.float32)
        Fre_inputs_tmp = Fre_inputs.to(torch.float32)
        Time_inputs = Time_inputs.to(torch.float32)
        # ================channel attention
        # b, 5, c -> b, 1, c, 5 -> b 5 c -> b 5*c+20 1
        Fre_inputs = Fre_inputs[:,:5,:].permute(0, 2, 1).unsqueeze(dim=1)
        fre_channel_att = self.F_channel_atten(Fre_inputs)
        Fre_inputs = rearrange(fre_channel_att, 'b o c f -> b o (c f)')
        Fre_inputs_tmp = Fre_inputs_tmp.to('cuda:0')
        Fre_inputs = torch.cat((Fre_inputs, Fre_inputs_tmp[:, 5, :20].unsqueeze(1)), dim=2)
        Fre_inputs = Fre_inputs.permute(0, 2, 1)
        # b, t, c -> b, 1, c, t -> b t c

        # Time_inputs = Time_inputs.permute(0, 2, 1).unsqueeze(dim=1)
        # T_channel_att = self.T_channel_atten(Time_inputs)
        # Time_inputs = T_channel_att.squeeze(dim=1).permute(0, 2, 1)
        #argon2:$argon2id$v=19$m=10240,t=10,p=8$V076JjR+SemTzIzePgBc1w$Y06pFdA6uMgxvht3IEeD+A
        # ================end channel attention
        # Fre
        # Fre_inputs = Fre_inputs.unsqueeze(2)
        Fre_inputs = self.Fre_src_emb(Fre_inputs)
        # Fre_inputs=self.conv(Fre_inputs).squeeze()
        # add class_token
        Fre_inputs=self.lear_f(Fre_inputs)
        b, n, _ = Fre_inputs.shape

        cls_tokens = repeat(self.Fre_cls_token, '() n e -> b n e', b=b)
        Fre_inputs = torch.cat((cls_tokens, Fre_inputs), dim=1)
        # Position embedding
        # Fre_inputs += self.Fre_pos_emb[:, :(n + 1)]
        Fre_inputs += self.Fre_pos_emb[:, :(n + 4)]

        # Fre_inputs += self.Fre_pos_emb[:, :n]
        Fre_pos_inputs = self.dropout(Fre_inputs)#+cls_tokens
        # Time
        #b, n, _ = Time_inputs.shape
        # conv_time=Time_inputs.unsqueeze(1)
        # conv_time=self.conv(conv_time)
        conv_time=self.lear_t(Time_inputs)



        #conv_time=rearrange(conv_time,'b c t f-> b (c t) f')
        # conv_time=conv_time.squeeze()
        b, n, _ = conv_time.shape
        cls_tokens = repeat(self.Time_cls_token, '() n e -> b n e', b=b)
        Time_inputs = torch.cat((cls_tokens, conv_time), dim=1)
        # Time_inputs += self.Time_pos_emb[:, :(n+1)]

        Time_inputs += self.Time_pos_emb[:, :(n + 4)]
        #Time_inputs += self.Time_pos_emb[:, :n]
        Time_pos_inputs = self.dropout(Time_inputs)
        # Time_inputs = self.Time_src_emb(Time_inputs)
        # cls_tokens = repeat(self.Time_cls_token, '() n e -> b n e', b=b)
        # Time_inputs = torch.cat((cls_tokens, Time_inputs), dim=1)
        # Time_inputs += self.Time_pos_emb[:, :(n+1)]
        # #Time_inputs += self.Time_pos_emb[:, :n]
        # Time_pos_inputs = self.dropout(Time_inputs)


        enc_self_attn_mask = None
        enc_self_attns = []

        # enc_outputs = torch.cat((Fre_pos_inputs, Time_pos_inputs), dim=1)

        for layer in self.layers:
            # enc_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            enc_outputs, enc_self_attn = layer(Fre_pos_inputs, Time_pos_inputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns