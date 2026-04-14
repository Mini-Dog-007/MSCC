# MSCC
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Autoformer_EncDec import series_decomp
from layers.Embed import DataEmbedding_wo_pos
from layers.StandardNorm import Normalize
from thop import profile
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class Mnmoudel(nn.Module):
    def __init__(self, configs ,num_layers=2):
        super(Mnmoudel, self).__init__()
        self.gelu = nn.GELU()
        self.num_layers = num_layers
        self.layer_norm = nn.LayerNorm(configs.d_model) 
        self.dropout = nn.Dropout(configs.dropout) 
        self.act = nn.GELU()
        self.sigmoid = nn.Sigmoid()
        self.conv1 = nn.ModuleList([
            torch.nn.Conv1d(in_channels=512, out_channels=512, kernel_size=5,padding=2)
            for i in range(num_layers)
        ])
        self.recursion = nn.Sequential(
            nn.Linear(in_features=512, out_features=1024),  
            nn.GELU(), 
            nn.Linear(in_features=1024, out_features=512), 
        )
    def forward(self,x , weight):
        gelu_out = self.gelu(weight)
        x = x.permute(0, 2, 1) 
        x = self.layer_norm(x)  
        x = x.permute(0, 2, 1)  
        for conv in self.conv1:
            conv_out = conv(x)
        x = conv_out
        x= self.act(x)
        x = x + gelu_out
        x = x.to(torch.float32)  
        return x

class MultiScaleSeasonMixing(nn.Module):
    """
    Bottom-up mixing season pattern
    """
    def  __init__(self, configs):
        super(MultiScaleSeasonMixing, self).__init__()
        self.down_sampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(  
                    torch.nn.Linear( 
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                    ),
                    nn.GELU(), 
                    torch.nn.Linear( 
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                    ),
                )
                for i in range(configs.down_sampling_layers)  
            ]
        )

    def forward(self, season_list):
        out_high = season_list[0]  
        out_low = season_list[1]   
        out_season_list = [out_high.permute(0, 2, 1)] 
        for i in range(len(season_list) - 1):
            out_low_res = self.down_sampling_layers[i](out_high)  
            out_low_res = self.high_pass_filter(out_low_res)
            out_low = out_low + out_low_res  
            out_high = out_low  
            if i + 2 <= len(season_list) - 1:
                out_low = season_list[i + 2] 
            out_season_list.append(out_high.permute(0, 2, 1)) 
        return out_season_list  

    def low_pass_filter(self, x, cutoff_ratio=0.25):
        B, C, T = x.shape
        freq = torch.fft.rfft(x, dim=-1) 
        cutoff = int((T // 2 + 1) * cutoff_ratio)
        mask = torch.zeros_like(freq)
        mask[:, :, :cutoff] = 1.0
        freq_filtered = freq * mask
        x_low = torch.fft.irfft(freq_filtered, n=T, dim=-1)
        return x_low

    def high_pass_filter(self, x, cutoff_ratio=0.25):
        B, C, T = x.shape
        freq = torch.fft.rfft(x, dim=-1)
        cutoff = int((T // 2 + 1) * cutoff_ratio)
        mask = torch.ones_like(freq)
        mask[:, :, :cutoff] = 0.0
        freq_filtered = freq * mask
        x_high = torch.fft.irfft(freq_filtered, n=T, dim=-1)
        return x_high

class MultiScaleTrendMixing(nn.Module):
    """
    Top-down mixing trend pattern
    """
    def __init__(self, configs):
        super(MultiScaleTrendMixing, self).__init__()

        self.attention = nn.MultiheadAttention(embed_dim=configs.d_model, num_heads=1, batch_first=True)
        self.conv = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3,
                              padding=1)
        self.up_sampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                        configs.seq_len // (configs.down_sampling_window ** i),
                    ),
                    nn.GELU(),
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.seq_len // (configs.down_sampling_window ** i),
                    ),
                )
                for i in reversed(range(configs.down_sampling_layers))
            ])

    def forward(self, trend_list):
        trend_list_reverse = trend_list.copy()
        trend_list_reverse.reverse()
        out_low = trend_list_reverse[0]
        out_high = trend_list_reverse[1]
        out_trend_list = [out_low.permute(0, 2, 1)]

        for i in range(len(trend_list_reverse) - 1):
            out_high_res = self.up_sampling_layers[i](out_low)
            out_high_res = self.high_pass_filter(out_high_res)
            out_high = out_high + out_high_res
            out_low = out_high
            if i + 2 <= len(trend_list_reverse) - 1:
                out_high = trend_list_reverse[i + 2]
            out_trend_list.append(out_low.permute(0, 2, 1))
        out_trend_list.reverse()
        return out_trend_list

    def low_pass_filter(self, x, cutoff_ratio=0.25):
        B, C, T = x.shape
        freq = torch.fft.rfft(x, dim=-1)  
        cutoff = int((T // 2 + 1) * cutoff_ratio)
        mask = torch.zeros_like(freq)
        mask[:, :, :cutoff] = 1.0
        freq_filtered = freq * mask
        x_low = torch.fft.irfft(freq_filtered, n=T, dim=-1)
        return x_low

    def high_pass_filter(self, x, cutoff_ratio=0.25):
        B, C, T = x.shape
        freq = torch.fft.rfft(x, dim=-1)
        cutoff = int((T // 2 + 1) * cutoff_ratio)
        mask = torch.ones_like(freq)
        mask[:, :, :cutoff] = 0.0
        freq_filtered = freq * mask
        x_high = torch.fft.irfft(freq_filtered, n=T, dim=-1)
        return x_high

class FrequencyDecomp(nn.Module):
    """
    Top-down mixing trend pattern
    """
    def __init__(self, configs):
        super(FrequencyDecomp, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=configs.d_model, num_heads=1, batch_first=True)

        self.conv = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3,
                              padding=1)
        self.up_sampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                        configs.seq_len // (configs.down_sampling_window ** i),
                    ),
                    nn.GELU(),
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.seq_len // (configs.down_sampling_window ** i),
                    ),
                )
                for i in reversed(range(configs.down_sampling_layers))
            ])

    def forward(self, trend_list):
        trend_list_reverse = trend_list.copy()
        trend_list_reverse.reverse()
        out_low = trend_list_reverse[0]
        out_high = trend_list_reverse[1]
        out_trend_list = [out_low.permute(0, 2, 1)]
        for i in range(len(trend_list_reverse) - 1):
            out_high_res = self.up_sampling_layers[i](out_low)
            out_high_res = self.low_pass_filter(out_high_res)
            out_high = out_high + out_high_res
            out_low = out_high
            if i + 2 <= len(trend_list_reverse) - 1:
                out_high = trend_list_reverse[i + 2]
            out_trend_list.append(out_low.permute(0, 2, 1))

        out_trend_list.reverse()

        return out_trend_list
    def low_pass_filter(self, x, cutoff_ratio=0.25):
        B, C, T = x.shape
        freq = torch.fft.rfft(x, dim=-1) 
        cutoff = int((T // 2 + 1) * cutoff_ratio)
        mask = torch.zeros_like(freq)
        mask[:, :, :cutoff] = 1.0
        freq_filtered = freq * mask

        x_low = torch.fft.irfft(freq_filtered, n=T, dim=-1)
        return x_low
    def high_pass_filter(self, x, cutoff_ratio=0.25):
        B, C, T = x.shape
        freq = torch.fft.rfft(x, dim=-1)
        cutoff = int((T // 2 + 1) * cutoff_ratio)

        mask = torch.ones_like(freq)
        mask[:, :, :cutoff] = 0.0
        freq_filtered = freq * mask

        x_high = torch.fft.irfft(freq_filtered, n=T, dim=-1)
        return x_high


class MultiCrossComponentFFT(nn.Module):
    def __init__(self, configs):
        super(MultiCrossComponentFFT, self).__init__()
        self.seq_len = configs.seq_len  
        self.pred_len = configs.pred_len 
        self.down_sampling_window = configs.down_sampling_window  
        self.layer_norm = nn.LayerNorm(configs.d_model) 
        self.dropout = nn.Dropout(configs.dropout)  
        self.channel_independence = configs.channel_independence 
        self.layer = configs.e_layers
        self.act = nn.GELU()
        if configs.decomp_method == 'moving_avg':
            self.decompsition = series_decomp(configs.moving_avg)    
        else:
            raise ValueError('decompsition is error')  
        if configs.channel_independence == 0:
            self.cross_layer = nn.Sequential(
                nn.Linear(in_features=configs.d_model, out_features=configs.d_ff),
                nn.GELU(),  
                nn.Linear(in_features=configs.d_ff, out_features=configs.d_model), 
            )
        self.mixing_multi_scale_season = MultiScaleSeasonMixing(configs)

        self.res_blocks = nn.ModuleList([FrequencyDecomp(configs)
                                         for _ in range(configs.e_layers)])
        self.mixing_multi_scale_trend = MultiScaleTrendMixing(configs)
        self.out_cross_layer = nn.Sequential(
            nn.Linear(in_features=configs.d_model, out_features=configs.d_ff),
            nn.GELU(),
            nn.Linear(in_features=configs.d_ff, out_features=configs.d_model),
        )
        self.recursion = nn.Sequential(
            nn.Linear(in_features=512, out_features=1024),  
            nn.GELU(), 
            nn.Linear(in_features=1024, out_features=512),  
        )
        self.conv_weight = Mnmoudel(configs, num_layers=1)


    def forward(self, x_list):
        length_list = []  
        for x in x_list:
            _, T, _ = x.size()  
            length_list.append(T)  

        season_list = []  
        trend_list = []  
        for x in x_list:
            season, trend = self.decompsition(x)  
            season = self.out_cross_layer(season)  
            trend = self.out_cross_layer(trend)  
            season_list.append(season.permute(0, 2, 1)) 
            trend_list.append(trend.permute(0, 2, 1))  

        out_list_season = []
        for ori, out_season, out_trend, length in zip(x_list, season_list, trend_list, length_list):
            out = self.conv_weight(out_season, out_trend)
            out_season_residual = out_season + out
            out_list_season.append(out_season_residual) 

        out_list_trend = []
        for ori, out_season, out_trend, length in zip(x_list, season_list, trend_list, length_list):
            out = self.conv_weight(out_trend, out_season)
            out_trend_residual = out_trend + out
            out_list_trend.append(out_trend_residual)

        season_list_fenjie_1 = [] 
        trend_list_fenjie_1 = []  
        for x in out_list_season :
            season, trend = self.decompsition(x.permute(0,2,1)) 
            season = self.out_cross_layer(season)  
            trend = self.out_cross_layer(trend)  
            season_list_fenjie_1.append(season.permute(0, 2, 1))  
            trend_list_fenjie_1 .append(trend.permute(0, 2, 1))  
        out_season_list_1 = self.mixing_multi_scale_season( season_list_fenjie_1)
        out_trend_list_1 = self.mixing_multi_scale_trend(trend_list_fenjie_1)
        out_list_1 = []  
        for ori, out_season, out_trend, length in zip(x_list, out_season_list_1 , out_trend_list_1, length_list):
      
            out = out_season + out_trend 
            if self.channel_independence:
                out = ori + self.out_cross_layer(out)  
            out_list_1.append(out[:, :length, :]) 

        season_list_fenjie_2= []  
        trend_list_fenjie_2 = [] 
        for x in out_list_trend:
            season, trend = self.decompsition(x.permute(0,2,1)) 
            season = self.out_cross_layer(season) 
            trend = self.out_cross_layer(trend)  
            season_list_fenjie_2.append(season.permute(0, 2, 1)) 
            trend_list_fenjie_2.append(trend.permute(0, 2, 1))  

        for i in range(self.layer):
            out_season_list_2 = self.res_blocks[i](season_list_fenjie_2)
        for i in range(self.layer):
            out_trend_list_2 = self.res_blocks[i](trend_list_fenjie_2)

        out_list_2 = [] 
        for ori, out_season, out_trend, length in zip(x_list,   out_season_list_2, out_trend_list_2, length_list):
            out = out_season + out_trend  
            if self.channel_independence:
                out = ori + self.out_cross_layer(out)  
            out_list_2.append(out[:, :length, :])  
        out_list = []  
        for ori, out_season, out_trend, length in zip(x_list, out_list_1, out_list_2, length_list):
            out = out_season + out_trend 
            if self.channel_independence:
                out = ori + self.out_cross_layer(out) 
            out_list.append(out[:, :length, :]) 

        return out_list  

class MultiScaleFusion(nn.Module):
    def __init__(self, input_dim=4, num_scales=3, hidden_dim=64, output_dim=4):
        super(MultiScaleFusion, self).__init__()
        self.num_scales = num_scales
        self.input_dim = input_dim

        self.mlp = nn.Sequential(
            nn.Linear(input_dim * num_scales, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x_list):
        """
        x_list: List of tensors, each with shape [B, 1, C]
        """
        x = torch.cat(x_list, dim=1)
        x = x.view(x.size(0), -1)
        out = self.mlp(x)
        out = out.unsqueeze(1)
        return out

class Channel_fusion(nn.Module):
    def __init__(self, mid_size=2048, act_fn='sigmoid'):
        super(Channel_fusion, self).__init__()
        self.mid_size = mid_size
        self.act_fn = getattr(torch, act_fn) if hasattr(torch, act_fn) else torch.sigmoid
        self.initialized = False  

    def _init_layers(self, in_dim, device):
        self.fc1 = nn.Linear(in_dim, self.mid_size).to(device) 
        self.fc2 = nn.Linear(self.mid_size, in_dim).to(device) 
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        self.initialized = True

    def forward(self, x):
        B, T, C = x.shape       
        if not self.initialized:
            self._init_layers(C, x.device)
        y = self.fc1(x)         
        y = self.act_fn(y)           
        y = self.fc2(y)            
        return y

class ChannelFusion_2(nn.Module):
    def __init__(self, mid_size=2048, act_fn='sigmoid'):
        super(ChannelFusion_2, self).__init__()
        in_dim =  512   # T * D
        self.fc1 = nn.Linear(in_dim, mid_size)  
        self.fc2 = nn.Linear(mid_size, in_dim)  

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

        if act_fn == 'sigmoid':
            self.act_fn = torch.sigmoid
        elif act_fn == 'relu':
            self.act_fn = F.relu
        elif act_fn == 'tanh':
            self.act_fn = torch.tanh
        else:
            raise ValueError(f"Unsupported activation: {act_fn}")

    def forward(self, x):
        B, T, C = x.shape
        y = self.fc1(x)           
        y = self.act_fn(y)            
        y = self.fc2(y)               
  
        return y

class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.fc = nn.Linear(2048, out_features=4)  
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.down_sampling_window = configs.down_sampling_window
        self.channel_independence = configs.channel_independence
        self.mccfft_blocks = nn.ModuleList([MultiCrossComponentFFT(configs)
                                         for _ in range(configs.e_layers)])

        self.preprocess = series_decomp(configs.moving_avg)
        self.enc_in = configs.enc_in
        self.use_future_temporal_feature = configs.use_future_temporal_feature

        if self.channel_independence == 1:
            self.enc_embedding = DataEmbedding_wo_pos(1, configs.d_model, configs.embed, configs.freq,
                                                      configs.dropout)
        else:
            self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                      configs.dropout)
        self.layer = configs.e_layers
        self.normalize_layers = torch.nn.ModuleList(
            [
                Normalize(self.configs.enc_in, affine=True, non_norm=True if configs.use_norm == 0 else False)
                for i in range(configs.down_sampling_layers + 1)
            ]
        )
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.predict_layers = torch.nn.ModuleList(
                [
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.pred_len,
                    )
                    for i in range(configs.down_sampling_layers + 1)
                ]
            )
            self.predict_layers = torch.nn.ModuleList([
                torch.nn.Sequential(
                    torch.nn.Linear(configs.seq_len // (configs.down_sampling_window ** i), 128),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(0.1),
                    torch.nn.Linear(128, configs.pred_len)
                )
                for i in range(configs.down_sampling_layers + 1)
            ])

            if self.channel_independence == 1:
                self.projection_layer = nn.Linear(
                    configs.d_model, 1, bias=True)
            else:
                self.projection_layer = nn.Linear(
                    configs.d_model, configs.c_out, bias=True)

                self.out_res_layers = torch.nn.ModuleList([
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.seq_len // (configs.down_sampling_window ** i),
                    )
                    for i in range(configs.down_sampling_layers + 1)
                ])

                self.regression_layers = torch.nn.ModuleList(
                    [
                        torch.nn.Linear(
                            configs.seq_len // (configs.down_sampling_window ** i),
                            configs.pred_len,
                        )
                        for i in range(configs.down_sampling_layers + 1)
                    ]
                )
        self.fusion_1 = Channel_fusion(mid_size=2048, act_fn='sigmoid').cuda()
        self.fusion_2 = ChannelFusion_2(mid_size=2048, act_fn='sigmoid')

    def out_projection(self, dec_out, i, out_res):
        dec_out = self.projection_layer(dec_out)
        out_res = out_res.permute(0, 2, 1)
        out_res = self.out_res_layers[i](out_res)
        out_res = self.regression_layers[i](out_res).permute(0, 2, 1)
        dec_out = dec_out + out_res
        return dec_out

    def pre_enc(self, x_list):
        if self.channel_independence == 1:
            return (x_list, None)
        else:
            out1_list = []
            out2_list = []
            for x in x_list:
                x_1, x_2 = self.preprocess(x)
                out1_list.append(x_1)
                out2_list.append(x_2)
            return (out1_list, out2_list)

    def __multi_scale_process_inputs(self, x_enc, x_mark_enc):
        if self.configs.down_sampling_method == 'max':
            down_pool = torch.nn.MaxPool1d(self.configs.down_sampling_window, return_indices=False)
        elif self.configs.down_sampling_method == 'avg':
            down_pool = torch.nn.AvgPool1d(self.configs.down_sampling_window)
        elif self.configs.down_sampling_method == 'conv':
            padding = 1 if torch.__version__ >= '1.5.0' else 2
            down_pool = nn.Conv1d(in_channels=self.configs.enc_in, out_channels=self.configs.enc_in,
                                  kernel_size=3, padding=padding,
                                  stride=self.configs.down_sampling_window,
                                  padding_mode='circular',
                                  bias=False)
        else:
            return x_enc, x_mark_enc
  
        x_enc = x_enc.permute(0, 2, 1)  
        x_enc_ori = x_enc 
        x_mark_enc_mark_ori = x_mark_enc
        x_enc_sampling_list = []  
        x_mark_sampling_list = []  
        x_enc_sampling_list.append(x_enc.permute(0, 2, 1)) 
        x_mark_sampling_list.append(x_mark_enc)

        for i in range(self.configs.down_sampling_layers):
            x_enc_sampling = down_pool(x_enc_ori) 

            x_enc_sampling_list.append(x_enc_sampling.permute(0, 2, 1))  
            x_enc_ori = x_enc_sampling  
            if x_mark_enc_mark_ori is not None:
                x_mark_sampling_list.append(x_mark_enc_mark_ori[:, ::self.configs.down_sampling_window, :])
                x_mark_enc_mark_ori = x_mark_enc_mark_ori[:, ::self.configs.down_sampling_window,
                                      :] 
        x_enc = x_enc_sampling_list
        if x_mark_enc_mark_ori is not None:
            x_mark_enc = x_mark_sampling_list
        else:
            x_mark_enc = x_mark_enc
        return x_enc, x_mark_enc
   
    def forecast(self, x_enc_1, x_mark_enc_1, x_dec, x_mark_dec):
        x_enc, x_mark_enc = self.__multi_scale_process_inputs(x_enc_1, x_mark_enc_1)
        x_list = []
        x_mark_list = []
        if x_mark_enc is not None:
            for i, x, x_mark in zip(range(len(x_enc)), x_enc, x_mark_enc):
                B, T, N = x.size() 
                x = self.normalize_layers[i](x, 'norm')  
                if self.channel_independence == 1:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)  
                    x_mark = x_mark.repeat(N, 1, 1)
                x_list.append(x)
                x_mark_list.append(x_mark)
        else:
            for i, x in zip(range(len(x_enc)), x_enc, ):
                B, T, N = x.size()
                x = self.normalize_layers[i](x, 'norm')
                if self.channel_independence == 1:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                x_list.append(x)

        enc_out_list = []
        x_list = self.pre_enc(x_list) 
        if x_mark_enc is not None:
            for i, x, x_mark in zip(range(len(x_list[0])), x_list[0], x_mark_list):
                enc_out = self.enc_embedding(x, x_mark)  
                enc_out_list.append(enc_out)
        else:
            for i, x in zip(range(len(x_list[0])), x_list[0]):
                enc_out = self.enc_embedding(x, None)  
                enc_out_list.append(enc_out)

        for i in range(self.layer):
            enc_out_list = self.mccfft_blocks[i](enc_out_list)
        Y_1 = self.fusion_1(enc_out_list[0])
        Y_2 = self.fusion_2(enc_out_list[1])
        y_list = [Y_1, Y_2]
        dec_out_list = self.future_multi_mixing(B, y_list, x_list)
        dec_out = torch.stack(dec_out_list, dim=-1).sum(-1)
        dec_out = self.normalize_layers[0](dec_out, 'denorm')
        return dec_out

    def future_multi_mixing(self, B, enc_out_list, x_list):
        dec_out_list = [] 
        if self.channel_independence == 1:
            x_list = x_list[0]  
            for i, enc_out in zip(range(len(x_list)), enc_out_list): 
                dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(
                    0, 2, 1) 
                if self.use_future_temporal_feature:
                    dec_out = dec_out + self.x_mark_dec
                    dec_out = self.projection_layer(dec_out)
                else:
                    dec_out = self.projection_layer(dec_out)  
                dec_out = dec_out.reshape(B, self.configs.c_out, self.pred_len).permute(0, 2,
                                                                                        1).contiguous()  
                dec_out_list.append(dec_out)
        else:
            for i, enc_out, out_res in zip(range(len(x_list[0])), enc_out_list, x_list[1]):
                dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(
                    0, 2, 1) 
                dec_out = self.out_projection(dec_out, i, out_res)
                dec_out_list.append(dec_out)

            return dec_out_list

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)  
            return dec_out
