import logging
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple, Union

import torch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from packaging.version import parse as V
from typeguard import check_argument_types

from espnet2.asr.ctc import CTC
from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.postencoder.abs_postencoder import AbsPostEncoder
from espnet2.asr.preencoder.abs_preencoder import AbsPreEncoder
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.asr.transducer.error_calculator import ErrorCalculatorTransducer
from espnet2.asr_transducer.utils import get_transducer_task_io
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet.nets.e2e_asr_common import ErrorCalculator
from espnet.nets.pytorch_backend.nets_utils import th_accuracy
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import (  # noqa: H301
    LabelSmoothingLoss,
)

if V(torch.__version__) >= V("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield


ACTIVATIONS = {
    'relu': torch.nn.ReLU,
    'leaky_relu': torch.nn.LeakyReLU,
    'gelu': torch.nn.GELU,
}

class FCLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, activation='gelu'):
        super(FCLayer, self).__init__()
        in_channels = int(in_channels)
        out_channels = int(out_channels)

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.linear = torch.nn.Linear(in_channels, out_channels)
        self.ins = torch.nn.InstanceNorm1d(out_channels, affine=True)
        self.act = ACTIVATIONS[activation]()

    def forward(self, x):
        out = self.linear(x)
        out = out.transpose(1,2)
        out = self.ins(out)
        out = self.act(out)
        out = out.transpose(1,2)

        return out


class ConvLayer2(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1):
        super(ConvLayer2, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad1d(reflection_padding)
        self.conv1d = torch.nn.Conv1d(
            in_channels, out_channels, kernel_size, stride, groups=groups)

    def forward(self, x, pad=False):
        if pad:
            out = self.reflection_pad(x)
        else:
            out = x
        out = self.conv1d(out)
        return out


class ResidualBlock2(torch.nn.Module):
    """ResidualBlock
    introduced in: https://arxiv.org/abs/1512.03385
    recommended architecture: http://torch.ch/blog/2016/02/04/resnets.html
    """

    def __init__(self, channels, activation='gelu'):
        super(ResidualBlock2, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=2)
        self.in1 = nn.InstanceNorm1d(channels, affine=True)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=2)
        self.in2 = nn.InstanceNorm1d(channels, affine=True)
        self.reflection_pad = torch.nn.ReflectionPad1d(1)
        self.act = ACTIVATIONS[activation]()

    def forward(self, x):
        residual = x
        out = self.act(self.in1(self.conv1(self.reflection_pad(x))))
        out = self.in2(self.conv2(out))
        out = out + residual
        return out


class ViewMaker(torch.nn.Module):
    '''Viewmaker network that stochastically maps a multichannel 2D input to an output of the same size.'''
    def __init__(self, num_channels=80, distortion_budget=0.02, activation='gelu',
                clamp=True, frequency_domain=False, downsample_to=False, num_res_blocks=0, num_noise=0):
        '''Initialize the Viewmaker network.

        Args:
            num_channels: Number of channels in the input (e.g. 1 for speech, 3 for images)
                Input will have shape [batch_size, num_channels, height, width]
            distortion_budget: Distortion budget of the viewmaker (epsilon, in the paper).
                Controls how strong the perturbations can be.
            activation: The activation function used in the network ('relu' and 'leaky_relu' currently supported)
            clamp: Whether to clamp the outputs to [0, 1] (useful to ensure output is, e.g., a valid image)
            frequency_domain: Whether to apply perturbation (and distortion budget) in the frequency domain.
                This is useful for shifting the inductive bias of the viewmaker towards more global / textural views.
            downsample_to: Downsamples the image, applies viewmaker, then upsamples. Possibly useful for 
                higher-resolution inputs, but not evaluaed in the paper.
            num_res_blocks: Number of residual blocks to use in the network.
        '''
        super().__init__()

        self.num_channels = num_channels
        self.num_res_blocks = num_res_blocks
        self.activation = activation
        self.clamp = clamp
        self.frequency_domain = frequency_domain
        self.downsample_to = downsample_to
        self.distortion_budget = distortion_budget
        self.num_noise = num_noise
        self.act = ACTIVATIONS[activation]()

        # Initial convolution layers (+ 1 for noise filter)
        self.conv1 = ConvLayer2(self.num_channels + self.num_noise, \
                self.num_channels, kernel_size=2, stride=1)
        self.in1 = torch.nn.InstanceNorm1d(self.num_channels, affine=True)
        self.conv2 = ConvLayer2(self.num_channels, self.num_channels, kernel_size=2, stride=1)
        self.in2 = torch.nn.InstanceNorm1d(self.num_channels, affine=True)
        self.conv3 = ConvLayer2(self.num_channels, self.num_channels, kernel_size=2, stride=1)
        self.in3 = torch.nn.InstanceNorm1d(self.num_channels, affine=True)
        self.conv4 = ConvLayer2(self.num_channels, self.num_channels, kernel_size=2, stride=1)
        self.in4 = torch.nn.InstanceNorm1d(self.num_channels, affine=True)

        self.conv5 = ConvLayer2(self.num_channels, \
                self.num_channels, kernel_size=2, stride=1)
        self.ins5 = torch.nn.InstanceNorm1d(self.num_channels, affine=True)
        self.conv6 = ConvLayer2(self.num_channels, self.num_channels, kernel_size=2, stride=1)

    @staticmethod
    def zero_init(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            # actual 0 has symmetry problems
            init.normal_(m.weight.data, mean=0, std=1e-4)
            # init.constant_(m.weight.data, 0)
            init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm1d):
            pass

    def add_noise_channel(self, x, num=1, bound_multiplier=1):
        # bound_multiplier is a scalar or a 1D tensor of length batch_size
        batch_size = x.size(0)
        filter_size = x.size(-1)
        shp = (batch_size, num, filter_size)
        bound_multiplier = torch.tensor(bound_multiplier, device=x.device)
        noise = torch.rand(shp, device=x.device) * bound_multiplier.view(-1, 1, 1)
        #if x.dtype == 'torch.cuda.float16':
        #    print('fuck'*100)
        #noise.type(torch.cuda.float16)
        noise = noise.half()
        return torch.cat((x, noise), dim=1)

    def basic_net(self, y, num_res_blocks=0, bound_multiplier=1):
        y = self.act(self.in1(self.conv1(y)))
        y = self.act(self.in2(self.conv2(y, pad=True)))
        y = self.act(self.in3(self.conv3(y)))
        y = self.act(self.in4(self.conv4(y, pad=True)))

        # Features that could be useful for other auxilary layers / losses.
        # [batch_size, 128]
        features = y.clone().mean([-1, -2])
        y = self.act(self.ins5(self.conv5(y, pad=True)))
        y = self.conv6(y)

        return y, features

    def get_delta(self, y_pixels, eps=1e-4):
        '''Constrains the input perturbation by projecting it onto an L1 sphere'''
        distortion_budget = self.distortion_budget
        delta = torch.tanh(y_pixels) # Project to [-1, 1]
        avg_magnitude = delta.abs().mean([1,2], keepdim=True)
        max_magnitude = distortion_budget
        delta = delta * max_magnitude / (avg_magnitude + eps)
        return delta

    def get_delta2(self, y_pixels, padding_mask, eps=1e-4):
        '''Constrains the input perturbation by projecting it onto an L1 sphere'''
        if padding_mask is not None:
            padding_mask_ = torch.logical_not(padding_mask)
            padding_mask_ = padding_mask_.long().unsqueeze(2)
            y_pixels = y_pixels.transpose(1,2)
            y_pixels *= padding_mask_
            y_pixels = y_pixels.transpose(1,2)

        distortion_budget = self.distortion_budget
        delta = torch.tanh(y_pixels) # Project to [-1, 1]
        avg_magnitude = delta.abs().mean([1,2], keepdim=True)
        max_magnitude = distortion_budget
        delta = delta * max_magnitude / (avg_magnitude + eps)
        return delta

    def forward(self, x, padding_mask):
        x = x.transpose(1,2)
        if self.downsample_to:
            # Downsample.
            x_orig = x
            x = torch.nn.functional.interpolate(
                x, size=(self.downsample_to, self.downsample_to), mode='bilinear')
        y = x

        if self.frequency_domain and 0:
            # Input to viewmaker is in frequency domain, outputs frequency domain perturbation.
            # Uses the Discrete Cosine Transform.
            # shape still [batch_size, C, W, H]
            y = dct.dct_2d(y)

        y_pixels, features = self.basic_net(y, self.num_res_blocks, bound_multiplier=1)
        #delta = self.get_delta(y_pixels.clone())
        delta = self.get_delta2(y_pixels.clone(), padding_mask)

        # Additive perturbation
        #result = x + delta
        result = y_pixels

        delta = delta.transpose(1,2)
        result = result.transpose(1,2)

        return result, delta


class ESPnetASRModel(AbsESPnetModel):
    """CTC-attention hybrid Encoder-Decoder model"""

    def __init__(
        self,
        vocab_size: int,
        token_list: Union[Tuple[str, ...], List[str]],
        frontend: Optional[AbsFrontend],
        specaug: Optional[AbsSpecAug],
        normalize: Optional[AbsNormalize],
        preencoder: Optional[AbsPreEncoder],
        encoder: AbsEncoder,
        postencoder: Optional[AbsPostEncoder],
        decoder: AbsDecoder,
        ctc: CTC,
        joint_network: Optional[torch.nn.Module],
        ctc_weight: float = 0.5,
        interctc_weight: float = 0.0,
        ignore_id: int = -1,
        lsm_weight: float = 0.0,
        length_normalized_loss: bool = False,
        report_cer: bool = True,
        report_wer: bool = True,
        sym_space: str = "<space>",
        sym_blank: str = "<blank>",
        extract_feats_in_collect_stats: bool = True,
        pac = True,
    ):
        assert check_argument_types()
        assert 0.0 <= ctc_weight <= 1.0, ctc_weight
        assert 0.0 <= interctc_weight < 1.0, interctc_weight

        super().__init__()
        # note that eos is the same as sos (equivalent ID)
        self.blank_id = 0
        self.sos = vocab_size - 1
        self.eos = vocab_size - 1
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id
        self.ctc_weight = ctc_weight
        self.interctc_weight = interctc_weight
        self.token_list = token_list.copy()

        self.frontend = frontend
        self.specaug = specaug
        self.normalize = normalize
        self.preencoder = preencoder
        self.postencoder = postencoder
        self.encoder = encoder
        if pac is not None: 
            self.pac = ViewMaker()
            self.mse = torch.nn.MSELoss()
        else: self.pac = None


        if not hasattr(self.encoder, "interctc_use_conditioning"):
            self.encoder.interctc_use_conditioning = False
        if self.encoder.interctc_use_conditioning:
            self.encoder.conditioning_layer = torch.nn.Linear(
                vocab_size, self.encoder.output_size()
            )

        self.use_transducer_decoder = joint_network is not None

        self.error_calculator = None

        if self.use_transducer_decoder:
            from warprnnt_pytorch import RNNTLoss

            self.decoder = decoder
            self.joint_network = joint_network

            self.criterion_transducer = RNNTLoss(
                blank=self.blank_id,
                fastemit_lambda=0.0,
            )

            if report_cer or report_wer:
                self.error_calculator_trans = ErrorCalculatorTransducer(
                    decoder,
                    joint_network,
                    token_list,
                    sym_space,
                    sym_blank,
                    report_cer=report_cer,
                    report_wer=report_wer,
                )
            else:
                self.error_calculator_trans = None

                if self.ctc_weight != 0:
                    self.error_calculator = ErrorCalculator(
                        token_list, sym_space, sym_blank, report_cer, report_wer
                    )
        else:
            # we set self.decoder = None in the CTC mode since
            # self.decoder parameters were never used and PyTorch complained
            # and threw an Exception in the multi-GPU experiment.
            # thanks Jeff Farris for pointing out the issue.
            if ctc_weight == 1.0:
                self.decoder = None
            else:
                self.decoder = decoder

            self.criterion_att = LabelSmoothingLoss(
                size=vocab_size,
                padding_idx=ignore_id,
                smoothing=lsm_weight,
                normalize_length=length_normalized_loss,
            )

            if report_cer or report_wer:
                self.error_calculator = ErrorCalculator(
                    token_list, sym_space, sym_blank, report_cer, report_wer
                )

        if ctc_weight == 0.0:
            self.ctc = None
        else:
            self.ctc = ctc

        self.extract_feats_in_collect_stats = extract_feats_in_collect_stats

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        identity=False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Frontend + Encoder + Decoder + Calc loss

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
            text: (Batch, Length)
            text_lengths: (Batch,)
            kwargs: "utt_id" is among the input.
        """
        assert text_lengths.dim() == 1, text_lengths.shape
        # Check that batch_size is unified
        assert (
            speech.shape[0]
            == speech_lengths.shape[0]
            == text.shape[0]
            == text_lengths.shape[0]
        ), (speech.shape, speech_lengths.shape, text.shape, text_lengths.shape)
        batch_size = speech.shape[0]

        # for data-parallel
        text = text[:, : text_lengths.max()]

        # 1. Encoder
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)
        intermediate_outs = None
        if self.pac is not None:
            loss_mse = encoder_out[2]
            encoder_out_pac = encoder_out[1]
            encoder_out = encoder_out[0]
            if isinstance(encoder_out, tuple):
                loss_mse = encoder_out[2]
                
                intermediate_outs = encoder_out[1]
                encoder_out = encoder_out[0]
                
                intermediate_outs_pac = encoder_out_pac[1]
                encoder_out_pac = encoder_out_pac[0]

            if identity:
                stats = dict()
                stats["loss_mse"] = loss_mse.detach() if loss_mse is not None else None
                loss_mse, stats, weight = force_gatherable((loss_mse, stats, batch_size), loss_mse.device)
                return loss_mse, stats, weight

        else:
            if isinstance(encoder_out, tuple):
                intermediate_outs = encoder_out[1]
                encoder_out = encoder_out[0]

        loss_att, acc_att, cer_att, wer_att = None, None, None, None
        loss_ctc, cer_ctc = None, None
        loss_transducer, cer_transducer, wer_transducer = None, None, None
        stats = dict()

        # 1. CTC branch
        if self.ctc_weight != 0.0:
            loss_ctc, cer_ctc = self._calc_ctc_loss(
                encoder_out, encoder_out_lens, text, text_lengths
            )
            
            # Collect CTC branch stats
            stats["loss_ctc"] = loss_ctc.detach() if loss_ctc is not None else None
            stats["cer_ctc"] = cer_ctc
            
            if self.pac is not None:
                loss_ctc_pac, cer_ctc_pac = self._calc_ctc_loss(
                    encoder_out_pac, encoder_out_lens, text, text_lengths
                )
                stats["loss_mse"] = loss_mse.detach() if loss_mse is not None else None
                stats["loss_ctc_pac"] = loss_ctc_pac.detach() if loss_ctc_pac is not None else None
                stats["cer_ctc_pac"] = cer_ctc_pac

        # Intermediate CTC (optional)
        #TODO: pac version has not been implemented yet
        loss_interctc = 0.0
        if self.interctc_weight != 0.0 and intermediate_outs is not None:
            for layer_idx, intermediate_out in intermediate_outs:
                # we assume intermediate_out has the same length & padding
                # as those of encoder_out
                loss_ic, cer_ic = self._calc_ctc_loss(
                    intermediate_out, encoder_out_lens, text, text_lengths
                )
                loss_interctc = loss_interctc + loss_ic

                # Collect Intermedaite CTC stats
                stats["loss_interctc_layer{}".format(layer_idx)] = (
                    loss_ic.detach() if loss_ic is not None else None
                )
                stats["cer_interctc_layer{}".format(layer_idx)] = cer_ic

            loss_interctc = loss_interctc / len(intermediate_outs)

            # calculate whole encoder loss
            loss_ctc = (
                1 - self.interctc_weight
            ) * loss_ctc + self.interctc_weight * loss_interctc

        #TODO: pac version has not been implemented yet
        if self.use_transducer_decoder:
            # 2a. Transducer decoder branch
            (
                loss_transducer,
                cer_transducer,
                wer_transducer,
            ) = self._calc_transducer_loss(
                encoder_out,
                encoder_out_lens,
                text,
            )

            if loss_ctc is not None:
                loss = loss_transducer + (self.ctc_weight * loss_ctc)
            else:
                loss = loss_transducer

            # Collect Transducer branch stats
            stats["loss_transducer"] = (
                loss_transducer.detach() if loss_transducer is not None else None
            )
            stats["cer_transducer"] = cer_transducer
            stats["wer_transducer"] = wer_transducer

        else:
            # 2b. Attention decoder branch
            if self.ctc_weight != 1.0:
                loss_att, acc_att, cer_att, wer_att = self._calc_att_loss(
                    encoder_out, encoder_out_lens, text, text_lengths
                )

                if self.pac is not None:
                    loss_att_pac, acc_att_pac, cer_att_pac, wer_att_pac = self._calc_att_loss(
                        encoder_out_pac, encoder_out_lens, text, text_lengths
                    )

            # 3. CTC-Att loss definition
            if self.ctc_weight == 0.0:
                loss = loss_att if self.pac is None else loss_att+loss_att_pac
            elif self.ctc_weight == 1.0:
                loss = loss_ctc if self.pac is None else loss_ctc+loss_ctc_pac
            else:
                loss = self.ctc_weight * loss_ctc + (1 - self.ctc_weight) * loss_att
                if self.pac is not None:
                    loss_adv = self.ctc_weight * loss_ctc_pac + (1 - self.ctc_weight) * loss_att_pac
                    loss_ctc_pac_clone = loss_ctc_pac

            # Collect Attn branch stats
            stats["loss_att"] = loss_att.detach() if loss_att is not None else None
            stats["acc"] = acc_att
            stats["cer"] = cer_att
            stats["wer"] = wer_att

            if self.pac is not None:
                stats["loss_att_pac"] = loss_att_pac.detach() if loss_att_pac is not None else None
                stats["acc_pac"] = acc_att_pac
                stats["cer_pac"] = cer_att_pac
                stats["wer_pac"] = wer_att_pac
                # Collect total loss stats
                stats["loss_adv"] = loss_adv.detach()
                
        # Collect total loss stats
        stats["loss"] = loss.detach()

        if self.pac is not None:
            loss = loss + loss_adv
            # force_gatherable: to-device and to-tensor if scalar for DataParallel
            loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
            loss_mse = force_gatherable(loss_mse, loss_mse.device)
            loss_ctc_pac_clone = force_gatherable(loss_ctc_pac_clone, loss_ctc_pac_clone.device)
            return (loss, loss_mse, loss_ctc_pac_clone), stats, weight

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def collect_feats(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        if self.extract_feats_in_collect_stats:
            feats, feats_lengths = self._extract_feats(speech, speech_lengths)
            logging.info("what is this? collect_feats?")
        else:
            # Generate dummy stats if extract_feats_in_collect_stats is False
            logging.warning(
                "Generating dummy stats for feats and feats_lengths, "
                "because encoder_conf.extract_feats_in_collect_stats is "
                f"{self.extract_feats_in_collect_stats}"
            )
            feats, feats_lengths = speech, speech_lengths
        return {"feats": feats, "feats_lengths": feats_lengths}

    def encode(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Frontend + Encoder. Note that this method is used by asr_inference.py

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
        """
        with autocast(False):
            # 1. Extract feats
            feats, feats_lengths = self._extract_feats(speech, speech_lengths)
            if self.pac is not None:
                feats_pac, _, loss_mse = self._extract_feats_pac(speech, speech_lengths)

            # 2. Data augmentation
            if self.specaug is not None and self.training:
                feats, feats_lengths = self.specaug(feats, feats_lengths)
                if self.pac is not None:
                    feats_pac, _ = self.specaug(feats_pac, feats_lengths)
            # 3. Normalization for feature: e.g. Global-CMVN, Utterance-CMVN
            if self.normalize is not None:
                feats, feats_lengths = self.normalize(feats, feats_lengths)
                if self.pac is not None:
                    feats_pac, _ = self.normalize(feats_pac, feats_lengths)
        # Pre-encoder, e.g. used for raw input data
        if self.preencoder is not None:
            feats, feats_lengths = self.preencoder(feats, feats_lengths)
            if self.pac is not None:
                feats_pac, _ = self.preencoder(feats_pac, feats_lengths)

        # 4. Forward encoder
        # feats: (Batch, Length, Dim)
        # -> encoder_out: (Batch, Length2, Dim2)
        if self.encoder.interctc_use_conditioning:
            encoder_out, encoder_out_lens, _ = self.encoder(
                feats, feats_lengths, ctc=self.ctc
            )
            if self.pac is not None:
                encoder_out_pac, _, _ = self.encoder(
                    feats_pac, feats_lengths, ctc=self.ctc
                )
   
        else:
            encoder_out, encoder_out_lens, _ = self.encoder(feats, feats_lengths)
            if self.pac is not None:
                encoder_out_pac, _, _ = self.encoder(feats_pac, feats_lengths)
        intermediate_outs = None
        if isinstance(encoder_out, tuple):
            intermediate_outs = encoder_out[1]
            encoder_out = encoder_out[0]
        if self.pac is not None and isinstance(encoder_out_pac, tuple):
            intermediate_outs_pac = encoder_out_pac[1]
            encoder_out_pac = encoder_out_pac[0]

        # Post-encoder, e.g. NLU
        if self.postencoder is not None:
            encoder_out, encoder_out_lens = self.postencoder(
                encoder_out, encoder_out_lens
            )
            if self.pac is not None:
                encoder_out_pac, _ = self.postencoder(
                    encoder_out_pac, encoder_out_lens
                )

        assert encoder_out.size(0) == speech.size(0), (
            encoder_out.size(),
            speech.size(0),
        )
        assert encoder_out.size(1) <= encoder_out_lens.max(), (
            encoder_out.size(),
            encoder_out_lens.max(),
        )
        
        assert encoder_out_pac.size(0) == speech.size(0), (
            encoder_out_pac.size(),
            speech.size(0),
        )
        assert encoder_out_pac.size(1) <= encoder_out_lens.max(), (
            encoder_out_pac.size(),
            encoder_out_lens.max(),
        )
        
        if self.pac is not None:
            if intermediate_outs is not None:
                return ((encoder_out, encoder_out_pac), (intermediate_outs, intermediate_outs_pac), loss_mse), encoder_out_lens

            return (encoder_out, encoder_out_pac, loss_mse), encoder_out_lens
        else:
            if intermediate_outs is not None:
                return (encoder_out, intermediate_outs), encoder_out_lens

            return encoder_out, encoder_out_lens

    def _extract_feats(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert speech_lengths.dim() == 1, speech_lengths.shape

        # for data-parallel
        speech = speech[:, : speech_lengths.max()]

        if self.frontend is not None:
            # Frontend
            #  e.g. STFT and Feature extract
            #       data_loader may send time-domain signal in this case
            # speech (Batch, NSamples) -> feats: (Batch, NFrames, Dim)
            feats, feats_lengths = self.frontend(speech, speech_lengths)
        else:
            # No frontend and no feature extract
            feats, feats_lengths = speech, speech_lengths
        return feats, feats_lengths
    
    def _extract_feats_pac(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert speech_lengths.dim() == 1, speech_lengths.shape

        # for data-parallel
        speech = speech[:, : speech_lengths.max()]

        if self.frontend is not None:
            # Frontend
            #  e.g. STFT and Feature extract
            #       data_loader may send time-domain signal in this case
            # speech (Batch, NSamples) -> feats: (Batch, NFrames, Dim)
            feats_origin, feats_lengths = self.frontend(speech, speech_lengths)
            #feats_origin = feats_origin.transpose(1,2)
            feats, _ = self.pac(feats_origin, None)
            #feats = feats.transpose(1,2)
            loss = self.mse(feats.reshape(-1, 80), feats_origin.reshape(-1, 80))
        else:
            # No frontend and no feature extract
            feats, feats_lengths = speech, speech_lengths
            feats = self.pac(feats, None)
        return feats, feats_lengths, loss

    def nll(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ) -> torch.Tensor:
        """Compute negative log likelihood(nll) from transformer-decoder

        Normally, this function is called in batchify_nll.

        Args:
            encoder_out: (Batch, Length, Dim)
            encoder_out_lens: (Batch,)
            ys_pad: (Batch, Length)
            ys_pad_lens: (Batch,)
        """
        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
        ys_in_lens = ys_pad_lens + 1

        # 1. Forward decoder
        decoder_out, _ = self.decoder(
            encoder_out, encoder_out_lens, ys_in_pad, ys_in_lens
        )  # [batch, seqlen, dim]
        batch_size = decoder_out.size(0)
        decoder_num_class = decoder_out.size(2)
        # nll: negative log-likelihood
        nll = torch.nn.functional.cross_entropy(
            decoder_out.view(-1, decoder_num_class),
            ys_out_pad.view(-1),
            ignore_index=self.ignore_id,
            reduction="none",
        )
        nll = nll.view(batch_size, -1)
        nll = nll.sum(dim=1)
        assert nll.size(0) == batch_size
        return nll

    def batchify_nll(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
        batch_size: int = 100,
    ):
        """Compute negative log likelihood(nll) from transformer-decoder

        To avoid OOM, this fuction seperate the input into batches.
        Then call nll for each batch and combine and return results.
        Args:
            encoder_out: (Batch, Length, Dim)
            encoder_out_lens: (Batch,)
            ys_pad: (Batch, Length)
            ys_pad_lens: (Batch,)
            batch_size: int, samples each batch contain when computing nll,
                        you may change this to avoid OOM or increase
                        GPU memory usage
        """
        total_num = encoder_out.size(0)
        if total_num <= batch_size:
            nll = self.nll(encoder_out, encoder_out_lens, ys_pad, ys_pad_lens)
        else:
            nll = []
            start_idx = 0
            while True:
                end_idx = min(start_idx + batch_size, total_num)
                batch_encoder_out = encoder_out[start_idx:end_idx, :, :]
                batch_encoder_out_lens = encoder_out_lens[start_idx:end_idx]
                batch_ys_pad = ys_pad[start_idx:end_idx, :]
                batch_ys_pad_lens = ys_pad_lens[start_idx:end_idx]
                batch_nll = self.nll(
                    batch_encoder_out,
                    batch_encoder_out_lens,
                    batch_ys_pad,
                    batch_ys_pad_lens,
                )
                nll.append(batch_nll)
                start_idx = end_idx
                if start_idx == total_num:
                    break
            nll = torch.cat(nll)
        assert nll.size(0) == total_num
        return nll

    def _calc_att_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):
        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
        ys_in_lens = ys_pad_lens + 1

        # 1. Forward decoder
        decoder_out, _ = self.decoder(
            encoder_out, encoder_out_lens, ys_in_pad, ys_in_lens
        )

        # 2. Compute attention loss
        loss_att = self.criterion_att(decoder_out, ys_out_pad)
        acc_att = th_accuracy(
            decoder_out.view(-1, self.vocab_size),
            ys_out_pad,
            ignore_label=self.ignore_id,
        )

        # Compute cer/wer using attention-decoder
        if self.training or self.error_calculator is None:
            cer_att, wer_att = None, None
        else:
            ys_hat = decoder_out.argmax(dim=-1)
            cer_att, wer_att = self.error_calculator(ys_hat.cpu(), ys_pad.cpu())

        return loss_att, acc_att, cer_att, wer_att

    def _calc_ctc_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):
        # Calc CTC loss
        loss_ctc = self.ctc(encoder_out, encoder_out_lens, ys_pad, ys_pad_lens)

        # Calc CER using CTC
        cer_ctc = None
        if not self.training and self.error_calculator is not None:
            ys_hat = self.ctc.argmax(encoder_out).data
            cer_ctc = self.error_calculator(ys_hat.cpu(), ys_pad.cpu(), is_ctc=True)
        return loss_ctc, cer_ctc

    def _calc_transducer_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        labels: torch.Tensor,
    ):
        """Compute Transducer loss.

        Args:
            encoder_out: Encoder output sequences. (B, T, D_enc)
            encoder_out_lens: Encoder output sequences lengths. (B,)
            labels: Label ID sequences. (B, L)

        Return:
            loss_transducer: Transducer loss value.
            cer_transducer: Character error rate for Transducer.
            wer_transducer: Word Error Rate for Transducer.

        """
        decoder_in, target, t_len, u_len = get_transducer_task_io(
            labels,
            encoder_out_lens,
            ignore_id=self.ignore_id,
            blank_id=self.blank_id,
        )

        self.decoder.set_device(encoder_out.device)
        decoder_out = self.decoder(decoder_in)

        joint_out = self.joint_network(
            encoder_out.unsqueeze(2), decoder_out.unsqueeze(1)
        )

        loss_transducer = self.criterion_transducer(
            joint_out,
            target,
            t_len,
            u_len,
        )

        cer_transducer, wer_transducer = None, None
        if not self.training and self.error_calculator_trans is not None:
            cer_transducer, wer_transducer = self.error_calculator_trans(
                encoder_out, target
            )

        return loss_transducer, cer_transducer, wer_transducer
