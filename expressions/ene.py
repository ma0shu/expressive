from types import SimpleNamespace

import librosa
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
class EneLoader(ExpressionLoader):
    expression_name = "ene"
    expression_info = _l("Energy (curve)")
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
        utau_time, utau_ene, utau_features = get_wav_features(
            wav_path=self.utau_path,
        )
        ref_time, ref_ene, ref_features = get_wav_features(
            wav_path=self.ref_path,
        )

        # Align all sequences to a common MIDI tick time base
        # NOTICE: features from UTAU WAV are the reference, and those from Ref. WAV are the query
        ene_tick, (time_aligned_ref_ene, *_unused), *_unused = align_sequence_tick(
            query_time=ref_time,
            queries=(ref_ene, *ref_features),
            reference_time=utau_time,
            references=(utau_ene, *utau_features),
            tempo=self.tempo,
            align_radius=align_radius,
        )

        ene_val = get_expression_energy(time_aligned_ref_ene, smoothness, scaler, bias)

        self.expression_tick, self.expression_val = ene_tick, ene_val
        self.logger.info(_("Expression extraction complete."))
        return self.expression_tick, self.expression_val


def extract_wav_energy(wav_path):
    sr = librosa.get_samplerate(wav_path)
    y, _ = librosa.load(wav_path, sr=sr)
    
    # Use Spectral Centroid to simulate Vocal Effort / Energy (ENE)
    # Higher vocal effort typically results in more high frequency harmonics, 
    # thus shifting the spectral centroid higher.
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    time = librosa.times_like(centroid, sr=sr)
    
    return time, centroid


def get_wav_features(wav_path):
    feature_times = []  # List of time sequences(list of lists)
    feature_vals = []  # List of feature sequences(list of lists)

    # Extract spectral centroid
    ene_time, ene = extract_wav_energy(wav_path)
    feature_times += [ene_time]
    feature_vals += [ene]

    # Extract dynamics and trends to help DTW alignment
    ene_dynamics_trends = seq_dynamics_trends(ene)
    feature_times += [ene_time] * len(ene_dynamics_trends)
    feature_vals += list(ene_dynamics_trends)

    # Unified time and features
    wav_time, (wav_ene, *wav_features) = unify_sequence_time(
        seq_times=feature_times, seq_vals=feature_vals
    )
    return wav_time, wav_ene, wav_features


def get_expression_energy(time_aligned_ene, smoothness=6, scaler=1.0, bias=0):
    base_scaler = 10.0
    smoothed_ene = gaussian_filter1d_with_nan(
        base_scaler * zscore(time_aligned_ene),
        sigma=smoothness,
    )
    return scaler * smoothed_ene + bias
