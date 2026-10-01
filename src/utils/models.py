from dataclasses import dataclass, field
from scenarIA.src.models.climax.arch import ClimaXClimateBench
from scenarIA.src.models.CNN import CNNBase
from scenarIA.src.models.unet import UNet
from scenarIA.src.models.time_unet import time_UNet
from scenarIA.src.models.convlstm import ConvLSTM
from scenarIA.src.models.convgru import ConvGRU
from scenarIA.src.models.trajGRU import TrajGRUMultiLayer
from scenarIA.src.models.smaat_unet import SmaAt_UNet
from scenarIA.src.models.attention_unet import AttentionUNet


@dataclass
class Dims:
    var_names: list          # self.inputs (+ 'climatology' si add_clim_to_predictors)
    out_vars: list           # self.outputs
    seq_length: int
    predict_only_last: bool
    img_size: tuple          # (lat, lon)

    @property
    def in_channels(self): return len(self.var_names)
    @property
    def n_outputs(self): return len(self.out_vars)
    @property
    def out_seq_len(self): return 1 if self.predict_only_last else self.seq_length

MODEL_REGISTRY = {}

def register(name):
    def deco(fn):
        MODEL_REGISTRY[name] = fn
        return fn
    return deco

def _hidden_list(t):
    h = t['lstm_units']
    return h if isinstance(h, list) else [h]


@register('cnn-lstm')
def _cnn_lstm(t, d):
    return CNNBase(slider=d.seq_length, height=d.img_size[0], width=d.img_size[1],
                   channels=d.in_channels, output_seq_len=d.out_seq_len,
                   time_module_name='lstm', hidden_size=t['lstm_units'])

@register('cnn-gru')
def _cnn_gru(t, d):
    return CNNBase(slider=d.seq_length, height=d.img_size[0], width=d.img_size[1],
                   channels=d.in_channels, output_seq_len=d.out_seq_len,
                   time_module_name='gru', hidden_size=t['lstm_units'])

@register('unet')
def _unet(t, d):
    return UNet(in_channels=d.in_channels * d.seq_length,
                out_channels=d.n_outputs * d.out_seq_len,
                init_features=t['unet_features'])

@register('time-unet')
def _time_unet(t, d):
    return time_UNet(num_input_vars=d.in_channels, num_output_vars=d.n_outputs,
                     longitude=d.img_size[1], latitude=d.img_size[0],
                     activation_function=None, datamodule_config=None,
                     channels_last=True, seq_to_seq=not d.predict_only_last,
                     seq_len=d.seq_length)

@register('convlstm')
def _convlstm(t, d):
    h = _hidden_list(t)
    return ConvLSTM(input_dim=d.in_channels, hidden_dim=h, kernel_size=(3, 3),
                    num_layers=len(h), batch_first=True, bias=True, return_all_layers=False)

@register('convgru')
def _convgru(t, d):
    h = _hidden_list(t)
    return ConvGRU(input_size=tuple(d.img_size), input_dim=d.in_channels, hidden_dim=h,
                   kernel_size=(3, 3), num_layers=len(h), batch_first=True, bias=True,
                   return_all_layers=False)

@register('trajgru')
def _trajgru(t, d):
    return TrajGRUMultiLayer(input_size=tuple(d.img_size), input_dim=d.in_channels,
                             hidden_dim=t['lstm_units'], L=t['links'], num_layers=1,
                             batch_first=True, return_all_layers=False)

@register('smaat-unet')
def _smaat(t, d):
    return SmaAt_UNet(n_channels=d.in_channels * d.seq_length,
                      n_classes=d.n_outputs * d.out_seq_len,
                      in_features=t['unet_features'], bilinear=False)

@register('attention-unet')
def _attention(t, d):
    return AttentionUNet(img_ch=d.in_channels * d.seq_length,
                         output_ch=d.n_outputs * d.out_seq_len,
                         in_features=t['unet_features'])

@register('climax')
def _climax(t, d):
    assert d.predict_only_last, "ClimaX : un seul pas de temps en sortie"
    c = t['climax']
    return ClimaXClimateBench(
        default_vars=d.var_names, out_vars=d.out_vars[0],
        img_size=list(d.img_size), time_history=d.seq_length,
        patch_size=c.get('patch_size', 16), embed_dim=c.get('embed_dim', 1024),
        depth=c.get('depth', 8), decoder_depth=c.get('decoder_depth', 2),
        num_heads=c.get('num_heads', 16), mlp_ratio=c.get('mlp_ratio', 4.0),
        drop_path=c.get('drop_path', 0.1), drop_rate=c.get('drop_rate', 0.1),
        parallel_patch_embed=c.get('parallel_patch_embed', False),
        freeze_encoder=c.get('freeze_encoder', False))


def build_model(config: dict, dims: Dims):
    arch = config['train']['arch']
    if arch not in MODEL_REGISTRY:
        raise KeyError(f"Architecture inconnue '{arch}'. Disponibles : {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[arch](config['train'], dims).float()