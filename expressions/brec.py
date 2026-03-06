from types import SimpleNamespace

import librosa
import numpy as np
from scipy.stats import zscore

from .base import Args, ExpressionLoader, register_expression
from utils.i18n import _, _l
from utils.seqtool import (
    unify_sequence_time,
    align_sequence_tick,
    gaussian_filter1d_with_nan,
    seq_dynamics_trends,
)


@register_expression
class BrecLoader(ExpressionLoader):
    expression_name = "brec"
    expression_info = _l("Breathiness (curve)")
    args = SimpleNamespace(
        align_radius    = Args(name="align_radius", type=int  , default=1  , help=_l("**Radius** for the FastDTW alignment algorithm; larger values allow more flexible alignment but increase computation time")),  # noqa: E501
        smoothness      = Args(name="smoothness"  , type=int  , default=6  , help=_l("Controls the **smoothness** of the expression curve using Gaussian filtering. Higher values produce smoother curves but may lose fine detail")),  # noqa: E501
        scaler          = Args(name="scaler"      , type=float, default=1.0, help=_l("**Scaling factor** applied to the expression curve. Values >1 amplify the expression, =1 keeps original intensity, <1 reduces it")),  # noqa: E501
        bias            = Args(name="bias"        , type=int  , default=0  , help=_l("**Bias** offset added to the expression curve. Positive values shift the curve upward; negative values shift it downward")),  # noqa: E501
    )

    def get_expression(
        self,
        align_radius = args.align_radius.default,
        smoothness   = args.smoothness  .default,
        scaler       = args.scaler      .default,
        bias         = args.bias        .default,
    ):
        self.logger.info(_("Extracting expression..."))

        # Extract features from WAV files
        utau_time, utau_brec, utau_features = get_wav_features(
            wav_path=self.utau_path,
        )
        ref_time, ref_brec, ref_features = get_wav_features(
            wav_path=self.ref_path,
        )

        # Align all sequences to a common MIDI tick time base
        # NOTICE: features from UTAU WAV are the reference, and those from Ref. WAV are the query
        brec_tick, (time_aligned_ref_brec, *_unused), *_unused = align_sequence_tick(
            query_time=ref_time,
            queries=(ref_brec, *ref_features),
            reference_time=utau_time,
            references=(utau_brec, *utau_features),
            tempo=self.tempo,
            align_radius=align_radius,
        )

        brec_val = get_expression_breathiness(time_aligned_ref_brec, smoothness, scaler, bias)

        self.expression_tick, self.expression_val = brec_tick, brec_val
        self.logger.info(_("Expression extraction complete."))
        return self.expression_tick, self.expression_val


def extract_wav_breathiness(wav_path):
    sr = librosa.get_samplerate(wav_path)
    y, _ = librosa.load(wav_path, sr=sr)
    
    # Use harmonic-percussive source separation (HPSS) to extract noisy (breath) components
    # The margin helps strictly separate harmonics from percussive/noise signals
    y_harmonic, y_percussive = librosa.effects.hpss(y, margin=(1.0, 5.0))
    
    # Calculate the RMS energy of the percussive/noise component
    rms_noise = librosa.feature.rms(y=y_percussive)[0]
    brec_time = librosa.times_like(rms_noise, sr=sr)
    return brec_time, rms_noise


def get_wav_features(wav_path):
    feature_times = []  # List of time sequences(list of lists)
    feature_vals = []  # List of feature sequences(list of lists)

    # Extract breathiness (noise RMS)
    brec_time, brec = extract_wav_breathiness(wav_path)
    feature_times += [brec_time]
    feature_vals += [brec]

    # Extract dynamics and trends to help DTW alignment
    brec_dynamics_trends = seq_dynamics_trends(brec)
    feature_times += [brec_time] * len(brec_dynamics_trends)
    feature_vals += list(brec_dynamics_trends)

    # Unified time and features
    wav_time, (wav_brec, *wav_features) = unify_sequence_time(
        seq_times=feature_times, seq_vals=feature_vals
    )
    return wav_time, wav_brec, wav_features


def get_expression_breathiness(time_aligned_brec, smoothness=6, scaler=1.0, bias=0):
    base_scaler = 10.0
    smoothed_brec = gaussian_filter1d_with_nan(
        base_scaler * zscore(time_aligned_brec),
        sigma=smoothness,
    )
    return scaler * smoothed_brec + bias
