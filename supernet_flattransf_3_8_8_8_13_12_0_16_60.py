import math

import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange

from linear import Linear_


class TransNetV2Supernet(nn.Module):
    def __init__(self, D=1024,
                 use_many_hot_targets=True,
                 use_frame_similarity=True,
                 use_mean_pooling=None,
                 dropout_rate=0.5,
                 use_resnet_like_top=False,
                 frame_similarity_on_last_layer=False,
                 use_color_histograms=True,
                 chromosome=None):
        super(TransNetV2Supernet, self).__init__()

        self.reprocess_layer = (lambda x: x / 255.)

        # TODO Layer 1 - 6
        self.Layer_0_3 = DilatedDCNNV2(3, 16, multiplier=1)
        self.Layer_1_8 = DilatedDCNNV2ABC(16 * 4, 16, multiplier=4, n_dilation=5, st_type="A")
        self.Layer_2_8 = DilatedDCNNV2ABC(16 * 4, 32, multiplier=4, n_dilation=5, st_type="A")
        self.Layer_3_8 = DilatedDCNNV2ABC(32 * 4, 32, multiplier=4, n_dilation=5, st_type="A")
        self.Layer_4_13 = DilatedDCNNV2(32 * 4, 64, multiplier=3, n_dilation=5)
        self.Layer_5_12 = DilatedDCNNV2(64 * 4, 64, multiplier=2, n_dilation=5)
        self.Layer_6_0 = Attention1D(dim_in=256 * 3 * 6, dim_out=1024, num_heads=4,
            qkv_bias=False,
            attn_drop=0.,
            proj_drop=0.,
            with_cls_token=False,
            skip_conv_proj=True,
            n_layer=0
        )

        self.pool = torch.nn.AvgPool3d(kernel_size=(1, 2, 2))

        if use_frame_similarity is True and use_color_histograms is True:
            in_features = 4864
        elif use_frame_similarity is True and use_color_histograms is False:
            in_features = 4736
        elif use_frame_similarity is False and use_color_histograms is True:
            in_features = 4736
        else:
            in_features = 4608

        self.fc1_0 = Linear_(in_features=in_features, out_features=D, bias=True, act="ReLU")
        in_features += 1024
        self.fc1 = Linear_(in_features=in_features, out_features=D, bias=True, act="ReLU")
        self.cls_layer1 = Linear_(in_features=1024, out_features=1, bias=True, act="Identity")
        self.cls_layer2 = Linear_(in_features=1024, out_features=1, bias=True, act="Identity") \
            if use_many_hot_targets else None
        self.frame_sim_layer = FrameSimilarity(in_channels=448, inner_channels=101) if use_frame_similarity else None
        self.color_hist_layer = ColorHistograms(lookup_window=101, output_dim=128) if use_color_histograms else None
        self.use_mean_pooling = use_mean_pooling
        self.dropout = torch.nn.Dropout(p=1.0 - dropout_rate) if dropout_rate is not None else None
        self.frame_similarity_on_last_layer = frame_similarity_on_last_layer
        self.resnet_like_top = use_resnet_like_top
        if self.resnet_like_top:
            raise Exception("Position resnet_like_top 1: should not be here !!!")
        # self.logits_act = torch.nn.Sigmoid()  # TODO put final activation here

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                fan_out = m.weight.size(0)
                fan_in = m.weight.size(1)
                limit = math.sqrt(6.0 / (fan_in + fan_out))
                m.weight.data.uniform_(-limit, limit)
                if m.bias is not None:
                    m.bias.data.zero_()
            # TODO !!!!!! supernet的参数初始化必须在外边一并执行，不能在算子内部执行，否则seed失效 !!!!!!
            elif isinstance(m, nn.Conv3d):
                init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")

    def forward(self, inputs, **kwargs):
        x = self.reprocess_layer(inputs)

        if self.resnet_like_top:
            raise Exception("Position resnet_like_top 2: should not be here !!!")

        block_features = []
        shortcut = None
        # TODO Layer 1 ~ 6
        num_layers = 6
        for layer_index in range(num_layers):
            if layer_index == 0:
                op = self.Layer_0_3
            elif layer_index == 1:
                op = self.Layer_1_8
            elif layer_index == 2:
                op = self.Layer_2_8
            elif layer_index == 3:
                op = self.Layer_3_8
            elif layer_index == 4:
                op = self.Layer_4_13
            elif layer_index == 5:
                op = self.Layer_5_12

            x = op(x)
            if layer_index in [0, 2, 4]:
                shortcut = x
            else:
                x = shortcut + x
                x = self.pool(x)
                block_features.append(x)
        transf_x = self.Layer_6_0(x, t=x.size(1), h=3, w=6)

        if self.use_mean_pooling:
            x = torch.mean(x, dim=[3, 4])  # out is [BS, C, N]
        else:
            # TODO actually go here !
            x = x.permute(0, 2, 3, 4, 1)
            shape = [x.shape[0], x.shape[1], np.prod(x.shape[2:])]
            x = x.reshape(shape=shape)  # out is [BS, C, N * H * W]

        if self.frame_sim_layer is not None and not self.frame_similarity_on_last_layer:
            x = torch.cat([self.frame_sim_layer(block_features), x], dim=2)

        if self.color_hist_layer is not None:
            x = torch.cat([self.color_hist_layer(inputs), x], dim=2)
        
        if transf_x is not None:
            x = torch.cat([transf_x, x], dim=2)

            x = self.fc1(x)
        else:
            x = self.fc1_0(x)

        if self.dropout is not None:
            x = self.dropout(x)

        if self.frame_sim_layer is not None and self.frame_similarity_on_last_layer:
            x = torch.cat([self.frame_sim_layer(block_features), x], dim=2)

        one_hot = self.cls_layer1(x)
        # one_hot = self.logits_act(one_hot)

        many_hot = None
        if self.cls_layer2 is not None:
            many_hot = self.cls_layer2(x)
            # many_hot = self.logits_act(self.cls_layer2(x))

        if many_hot != None:
            return one_hot, many_hot
        return one_hot


def gather_nd(params, indices):
    """ The same as tf.gather_nd but batched gather is not supported yet.
    indices is an k-dimensional integer tensor, best thought of as a (k-1)-dimensional tensor of indices into params, where each element defines a slice of params:

    output[\\(i_0, ..., i_{k-2}\\)] = params[indices[\\(i_0, ..., i_{k-2}\\)]]

    Args:
        params (Tensor): "n" dimensions. shape: [x_0, x_1, x_2, ..., x_{n-1}]
        indices (Tensor): "k" dimensions. shape: [y_0,y_2,...,y_{k-2}, m]. m <= n.

    Returns: gathered Tensor.
        shape [y_0,y_2,...y_{k-2}] + params.shape[m:]

    """
    orig_shape = list(indices.shape)
    num_samples = np.prod(orig_shape[:-1])
    m = orig_shape[-1]
    n = len(params.shape)

    if m <= n:
        out_shape = orig_shape[:-1] + list(params.shape)[m:]
    else:
        raise ValueError(f'the last dimension of indices must less or equal to the rank of params. '
                         f'Got indices:{indices.shape}, params:{params.shape}. {m} > {n}')

    indices = indices.reshape((num_samples, m)).transpose(0, 1).tolist()
    output = params[indices]  # (num_samples, ...)
    return output.reshape(out_shape).contiguous()


class FrameSimilarity(nn.Module):

    def __init__(self,
                 in_channels,
                 inner_channels,
                 similarity_dim=128,
                 lookup_window=101,
                 output_dim=128,
                 stop_gradient=False,
                 use_bias=True):
        super(FrameSimilarity, self).__init__()

        self.projection = Linear_(in_features=in_channels, out_features=similarity_dim,
                                  bias=use_bias, act="Identity")
        self.fc = Linear_(in_features=inner_channels, out_features=output_dim, bias=True, act="ReLU")

        self.lookup_window = lookup_window
        self.stop_gradient = stop_gradient
        assert lookup_window % 2 == 1, "`lookup_window` must be odd integer"
        if torch.cuda.is_available() is True:
            self.device = "cuda"
        else:
            self.device = "cpu"

    def forward(self, inputs):
        # pt version [BS, C, N, H, W], so [3, 4] means apply avg on spatial dim, out dim is [BS, C, N]
        x = torch.cat([torch.mean(x, dim=[3, 4]) for x in inputs], dim=1)

        if self.stop_gradient:
            x = x.detach()

        x = x.permute(dims=[0, 2, 1])  # out is [BS, N ,C]
        batch_size, time_window, old_channels = x.shape
        x = x.reshape(shape=[batch_size * time_window, old_channels])  # [BS X N, C]
        x = self.projection(x)
        x = F.normalize(x, p=2, dim=1)  # norm at C dim

        _, new_channels = x.shape
        x = x.reshape(shape=[batch_size, time_window, new_channels])
        y = x.permute(dims=[0, 2, 1])
        similarities = torch.matmul(x, y)  # [batch_size, time_window, time_window]
        # note that it operates on dimensions of the input tensor in a backward fashion (from last dimension to the first dimension)
        similarities_padded = F.pad(similarities,
                                    pad=[(self.lookup_window - 1) // 2, (self.lookup_window - 1) // 2, 0, 0, 0, 0])

        batch_indices = torch.arange(0, batch_size, device=self.device). \
            reshape(shape=[batch_size, 1, 1]). \
            repeat([1, time_window, self.lookup_window])
        time_indices = torch.arange(0, time_window, device=self.device). \
            reshape(shape=[1, time_window, 1]). \
            repeat([batch_size, 1, self.lookup_window])
        lookup_indices = torch.arange(0, self.lookup_window, device=self.device). \
                             reshape(shape=[1, 1, self.lookup_window]). \
                             repeat([batch_size, time_window, 1]) + time_indices

        indices = torch.stack([batch_indices, time_indices, lookup_indices], dim=-1)

        similarities = gather_nd(similarities_padded, indices)
        return self.fc(similarities)


class ColorHistograms(nn.Module):

    def __init__(self, lookup_window=101, output_dim=128):
        super(ColorHistograms, self).__init__()
        self.fc = Linear_(in_features=101, out_features=output_dim, bias=True, act="ReLU") \
            if output_dim is not None else None
        self.lookup_window = lookup_window
        assert lookup_window % 2 == 1, "`lookup_window` must be odd integer"
        if torch.cuda.is_available() is True:
            self.device = "cuda"
        else:
            self.device = "cpu"

    def unsorted_segment_sum(self, data, segment_ids, num_segments):
        """
        https://gist.github.com/bbrighttaer/207dc03b178bbd0fef8d1c0c1390d4be
        """
        assert all([i in data.shape for i in segment_ids.shape]), "segment_ids.shape should be a prefix of data.shape"
        # segment_ids is a 1-D tensor repeat it to have the same shape as data
        if len(segment_ids.shape) == 1:
            s = torch.prod(torch.tensor(data.shape[1:])).long()
            segment_ids = segment_ids.repeat_interleave(s).view(segment_ids.shape[0], *data.shape[1:])
        assert data.shape == segment_ids.shape, "data.shape and segment_ids.shape should be equal"
        shape = [num_segments] + list(data.shape[1:])
        tensor = torch.zeros(*shape, device=self.device).scatter_add(0, segment_ids, data.float())
        tensor = tensor.type(data.dtype)
        return tensor

    def compute_color_histograms(self, frames):
        frames = frames.type(dtype=torch.int32)
        # pt version [BS, C, N, H, W]  ---> tf version [BS, N, H, W, C]
        frames = frames.permute(0, 2, 3, 4, 1)

        def get_bin(frames):
            # returns 0 .. 511
            R, G, B = frames[:, :, 0], frames[:, :, 1], frames[:, :, 2]
            R, G, B = R >> 5, G >> 5, B >> 5
            return (R << 6) + (G << 3) + B

        batch_size, time_window, height, width = frames.shape[0], frames.shape[1], frames.shape[2], frames.shape[3]

        no_channels = frames.shape[-1]
        assert no_channels == 3 or no_channels == 6
        if no_channels == 3:
            frames_flatten = frames.reshape(shape=[batch_size * time_window, height * width, 3])
        else:
            frames_flatten = frames.reshape(shape=[batch_size * time_window, height * width * 2, 3])

        binned_values = get_bin(frames_flatten)
        frame_bin_prefix = torch.arange(0, batch_size * time_window, dtype=torch.int32, device=self.device) << 9
        frame_bin_prefix = frame_bin_prefix.unsqueeze(dim=-1)
        binned_values = binned_values + frame_bin_prefix

        ones = torch.ones_like(binned_values, dtype=torch.int32, device=self.device)
        histograms = self.unsorted_segment_sum(data=ones,
                                               segment_ids=binned_values.type(dtype=torch.long),
                                               num_segments=batch_size * time_window * 512)
        histograms = torch.sum(histograms, dim=1)
        histograms = histograms.reshape(shape=[batch_size, time_window, 512])

        histograms_normalized = histograms.type(dtype=torch.float32)
        histograms_normalized = histograms_normalized / torch.norm(histograms_normalized, dim=2, keepdim=True)
        return histograms_normalized

    def forward(self, inputs):
        x = self.compute_color_histograms(inputs)

        batch_size, time_window = x.shape[0], x.shape[1]
        y = x.permute(dims=[0, 2, 1])
        similarities = torch.matmul(x, y)  # [batch_size, time_window, time_window]
        # note that it operates on dimensions of the input tensor in a backward fashion (from last dimension to the first dimension)
        similarities_padded = F.pad(similarities,
                                    pad=[(self.lookup_window - 1) // 2, (self.lookup_window - 1) // 2, 0, 0, 0, 0])

        batch_indices = torch.arange(0, batch_size, device=self.device). \
            reshape(shape=[batch_size, 1, 1]). \
            repeat([1, time_window, self.lookup_window])
        time_indices = torch.arange(0, time_window, device=self.device). \
            reshape(shape=[1, time_window, 1]). \
            repeat([batch_size, 1, self.lookup_window])
        lookup_indices = torch.arange(0, self.lookup_window, device=self.device). \
                             reshape(shape=[1, 1, self.lookup_window]). \
                             repeat([batch_size, time_window, 1]) + time_indices

        indices = torch.stack([batch_indices, time_indices, lookup_indices], dim=-1)

        similarities = gather_nd(similarities_padded, indices)

        if self.fc is not None:
            return self.fc(similarities)
        return similarities


class ConvexCombinationRegularization(nn.Module):

    def __init__(self, ):
        super(ConvexCombinationRegularization, self).__init__()

        raise Exception("ConvexCombinationRegularization: should not init this class !!!")

    def forward(self, inputs):
        pass


class DilatedDCNNV2ABC(nn.Module):
    def __init__(self, in_channels,
                 filters,
                 batch_norm=True,
                 activation=F.relu,
                 octave_conv=False,
                 multiplier=4,
                 n_dilation=4,
                 st_type="A"):
        super(DilatedDCNNV2ABC, self).__init__()
        assert not (octave_conv and batch_norm)

        self.share = torch.nn.Conv3d(in_channels=in_channels,
                                     out_channels=multiplier * filters,
                                     kernel_size=(1, 3, 3),
                                     padding=(0, 1, 1),
                                     dilation=(1, 1, 1),
                                     bias=False)
        init.kaiming_normal_(self.share.weight, mode="fan_in", nonlinearity="relu")

        self.conv_blocks = nn.ModuleList()

        n_in_plane = multiplier * filters
        if st_type == "B":
            n_in_plane = in_channels

        n_filter_per_module = (filters * 4) // n_dilation  # multiplier
        for dilation in range(n_dilation-1):
            self.conv_blocks.append(
                Conv3DConfigurable(
                    n_in_plane,
                    n_filter_per_module,
                    2 ** dilation,
                    mid_filter=n_in_plane,
                    separable=True,
                    sharable=True,
                    use_bias=not batch_norm,
                    octave=octave_conv
                )
            )
        self.conv_blocks.append(
            Conv3DConfigurable(
                n_in_plane,
                (filters * 4) - n_filter_per_module * (n_dilation-1),  # multiplier
                2 ** (n_dilation - 1),
                mid_filter=n_in_plane,
                separable=True,
                sharable=True,
                use_bias=not batch_norm,
                octave=octave_conv
            )
        )

        self.octave = octave_conv
        self.multiplier = multiplier
        self.n_dilation = n_dilation
        self.st_type = st_type

        self.batch_norm = torch.nn.BatchNorm3d(
            num_features=filters * 4,  # multiplier,
            eps=1e-3, momentum=0.1) if batch_norm else None
        self.activation = activation

    def forward(self, inputs):
        feature = self.share(inputs)
        if self.st_type == "A":
            x = []
#             print(len(self.conv_blocks), feature.size())
            for block in self.conv_blocks:
#                 print(block)
                x.append(block(feature))
            x = torch.cat(x, dim=1)
        elif self.st_type == "B":
            x = []
            for block in self.conv_blocks:
                x.append(block(inputs))
            x = torch.cat(x, dim=1)
            x = x + feature
        elif self.st_type == "C":
            x = []
            for block in self.conv_blocks:
                x.append(block(feature))
            x = torch.cat(x, dim=1)
            x = x + feature
        else:
            raise Exception("Not Implemented ST Type" + self.st_type)

        if self.octave:
            raise Exception("Position octave 1: should not be here !!!")

        if self.batch_norm is not None:
            x = self.batch_norm(x)

        if self.activation is not None:
            if self.octave:
                raise Exception("Position octave 2: should not be here !!!")
            else:
                x = self.activation(x)
        return x


class DilatedDCNNV2(nn.Module):
    def __init__(self, in_channels,
                 filters,
                 multiplier=2,
                 n_dilation=4,
                 batch_norm=True,
                 activation=F.relu,
                 octave_conv=False):
        super(DilatedDCNNV2, self).__init__()
        assert not (octave_conv and batch_norm)

        self.n_dilation = n_dilation
        self.conv_blocks = nn.ModuleList()


        n_filter_per_module = (filters * 4) // n_dilation  # multiplier
        for dilation in range(n_dilation-1):
            self.conv_blocks.append(
                Conv3DConfigurable(
                    in_channels,
                    n_filter_per_module,
                    mid_filter=multiplier*filters,
                    dilation_rate=2 ** dilation,
                    use_bias=not batch_norm,
                    octave=octave_conv
                )
            )
        self.conv_blocks.append(
            Conv3DConfigurable(
                in_channels,
                (filters * 4) - n_filter_per_module * (n_dilation-1),  # multiplier
                mid_filter=multiplier*filters,
                dilation_rate=2 ** (n_dilation-1),
                use_bias=not batch_norm,
                octave=octave_conv
            )
        )

        self.batch_norm = torch.nn.BatchNorm3d(num_features=filters * 4, eps=1e-3, momentum=0.1) if batch_norm else None
        self.activation = activation
        self.octave = octave_conv

    def forward(self, inputs):
        x = []
        for block in self.conv_blocks:
            x.append(block(inputs))
        x = torch.cat(x, dim=1)

        if self.octave:
            raise Exception("Position octave 1: should not be here !!!")


        if self.batch_norm is not None:
            x = self.batch_norm(x)

        if self.activation is not None:
            if self.octave:
                raise Exception("Position octave 2: should not be here !!!")
            else:
                x = self.activation(x)
        return x


class Conv3DConfigurable(nn.Module):
    def __init__(self, in_channels,
                 filters,
                 dilation_rate,
                 mid_filter=None,
                 separable=True,
                 sharable=False,
                 octave=False,
                 use_bias=False):
        super(Conv3DConfigurable, self).__init__()
        assert not (separable and octave)

        if separable:
            # (2+1)D convolution https://arxiv.org/pdf/1711.11248.pdf
            from torch.nn import init
            self.layers = nn.ModuleList()
            if not sharable:
                conv1 = torch.nn.Conv3d(in_channels=in_channels,
                                        out_channels=2 * filters if mid_filter is None else mid_filter,
                                        kernel_size=(1, 3, 3),
                                        padding=(0, 1, 1),
                                        dilation=(1, 1, 1),
                                        bias=False)
                init.kaiming_normal_(conv1.weight, mode="fan_in", nonlinearity="relu")
                self.layers.append(conv1)

            conv2 = torch.nn.Conv3d(in_channels=2 * filters if mid_filter is None else mid_filter,
                                    out_channels=filters,
                                    kernel_size=(3, 1, 1),
                                    padding=(1 * dilation_rate, 0, 0),
                                    dilation=(dilation_rate, 1, 1),
                                    bias=use_bias)
            init.kaiming_normal_(conv2.weight, mode="fan_in", nonlinearity="relu")
            self.layers.append(conv2)
        elif octave:
            raise Exception("Positon octave 3: should not be here !!!")
        else:
            raise Exception("Positon else 1: should not be here !!!")

    def forward(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x
    
    
class Attention1D(nn.Module):
    def __init__(self,
                 dim_in,
                 dim_out,
                 num_heads,
                 qkv_bias=False,
                 attn_drop=0.,
                 proj_drop=0.,
                 with_cls_token=False,
                 skip_conv_proj=False,
                 n_layer=1,
                 **kwargs
                 ):
        super().__init__()
        self.dim = dim_out
        self.num_heads = num_heads
        self.n_layer = n_layer
        # head_dim = self.qkv_dim // num_heads
        self.scale = dim_out ** -0.5
        self.with_cls_token = with_cls_token
        
        self.proj_q = nn.ModuleList()
        self.proj_k = nn.ModuleList()
        self.proj_v = nn.ModuleList()
        self.attn_drop = nn.ModuleList()
        self.proj = nn.ModuleList()
        self.proj_drop = nn.ModuleList()
        
        for _ in range(n_layer):
            self.proj_q.append(nn.Linear(dim_in, dim_out, bias=qkv_bias))
            self.proj_k.append(nn.Linear(dim_in, dim_out, bias=qkv_bias))
            self.proj_v.append(nn.Linear(dim_in, dim_out, bias=qkv_bias))

            self.attn_drop.append(nn.Dropout(attn_drop))
            self.proj.append(nn.Linear(dim_out, dim_out))
            self.proj_drop.append(nn.Dropout(proj_drop))
            
            dim_in = dim_out

    def forward(self, x, t, h, w):
        x = rearrange(x, 'b c t H W -> b t (c H W)')
        if self.n_layer == 0:
            return None
        
        for idx in range(self.n_layer):
            q = rearrange(self.proj_q[idx](x), 'b t (h d) -> b h t d', h=self.num_heads)
            k = rearrange(self.proj_k[idx](x), 'b t (h d) -> b h t d', h=self.num_heads)
            v = rearrange(self.proj_v[idx](x), 'b t (h d) -> b h t d', h=self.num_heads)

            attn_score = torch.einsum('bhlk,bhtk->bhlt', [q, k]) * self.scale
            attn = F.softmax(attn_score, dim=-1)
            attn = self.attn_drop[idx](attn)

            x = torch.einsum('bhlt,bhtv->bhlv', [attn, v])
            x = rearrange(x, 'b h t d -> b t (h d)')

            x = self.proj[idx](x)
            x = self.proj_drop[idx](x)
        return x