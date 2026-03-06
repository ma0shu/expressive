from types import SimpleNamespace

import numpy as np

from .base import Args, ExpressionLoader, register_expression
from utils.i18n import _, _l, _lf
from utils.seqtool import (
    unify_sequence_time,
    align_sequence_tick,
    gaussian_filter1d_with_nan,
)
from utils.log import StreamToLogger
from .pitd import extract_wav_frequency

@register_expression
class VoicLoader(ExpressionLoader):
    expression_name = "voic"
    expression_info = _l("Voicing (curve)")
    backend_choices = {
        "swift-f0": _l("fast, CPU-based (ONNX Runtime)"),
        "crepe": _l("classic but slow, CPU & NVIDIA GPU (TensorFlow)"),
    }
    
    args = SimpleNamespace(
        backend         = Args(name="backend"        , type=str  , default="swift-f0", choices=list(backend_choices.keys()), help=_lf("**F0 detection backend** for extracting voicing confidence. Available options:\n\n%s\n\n", lambda: "\n".join([f"- `{k}`: {v}" for k, v in VoicLoader.backend_choices.items()]))),  # noqa: E501
        align_radius    = Args(name="align_radius"   , type=int  , default=1   , help=_l("**Radius** for the FastDTW alignment algorithm; larger values allow more flexible alignment but increase computation time")),  # noqa: E501
        smoothness      = Args(name="smoothness"     , type=int  , default=2   , help=_l("Controls the **smoothness** of the expression curve using Gaussian filtering. Higher values produce smoother curves but may lose fine detail")),  # noqa: E501
        scaler          = Args(name="scaler"         , type=float, default=100.0, help=_l("**Scaling factor** applied to the expression curve. Values >1 amplify the expression, =1 keeps original intensity, <1 reduces it")),  # noqa: E501
        bias            = Args(name="bias"           , type=float, default=0.0 , help=_l("**Bias** offset added to the expression curve. Positive values shift the curve upward; negative values shift it downward")),  # noqa: E501
    )

    def get_expression(
        self,
        backend         = args.backend        .default,
        align_radius    = args.align_radius   .default,
        smoothness      = args.smoothness     .default,
        scaler          = args.scaler         .default,
        bias            = args.bias           .default,
    ):
        self.logger.info(_("Extracting expression..."))

        with StreamToLogger(self.logger, tee=True):
            utau_time, utau_conf, utau_features = get_wav_features(
                wav_path=self.utau_path, backend=backend
            )

        with StreamToLogger(self.logger, tee=True):
            ref_time, ref_conf, ref_features = get_wav_features(
                wav_path=self.ref_path, backend=backend
            )

        # Align all sequences to a common MIDI tick time base
        # NOTICE: features from UTAU WAV are the reference, and those from Ref. WAV are the query
        voic_tick, (time_aligned_ref_conf, *_unused), *_unused = align_sequence_tick(
            query_time=ref_time,
            queries=(ref_conf, *ref_features),
            reference_time=utau_time,
            references=(utau_conf, *utau_features),
            tempo=self.tempo,
            align_radius=align_radius,
        )

        voic_val = get_expression_voicing(time_aligned_ref_conf, smoothness, scaler, bias)

        self.expression_tick, self.expression_val = voic_tick, voic_val
        self.logger.info(_("Expression extraction complete."))
        return self.expression_tick, self.expression_val


def get_wav_features(wav_path, backend="swift-f0"):
    feature_times = []
    feature_vals = []

    # Use extract_wav_frequency from pitd to get confidence
    time, frequency, confidence = extract_wav_frequency(wav_path, backend=backend)
    
    conf = np.array(confidence)
    feature_times.append(time)
    feature_vals.append(conf)

    # Note: For alignment, pitd uses MFCCs and pitch dynamics as features.
    # To keep alignment robust, we can just use the confidence values themselves,
    # or rely solely on DTW over confidence. But usually more features (like RMS/MFCC)
    # produce better DTW paths. Let's use simple dynamics for confidence.
    from utils.seqtool import seq_dynamics_trends
    conf_features = seq_dynamics_trends(conf)
    feature_times += [time] * len(conf_features)
    feature_vals += list(conf_features)

    wav_time, (wav_conf, *wav_features) = unify_sequence_time(
        seq_times=feature_times, seq_vals=feature_vals
    )
    return wav_time, wav_conf, wav_features


def get_expression_voicing(time_aligned_conf, smoothness=2, scaler=100.0, bias=0.0):
    # Confidence is typically [0, 1]. Scaling by 100 maps it to [0, 100] standard OpenUtau range.
    smoothed_voic = gaussian_filter1d_with_nan(
        time_aligned_conf,
        sigma=smoothness,
    )
    return scaler * smoothed_voic + bias
