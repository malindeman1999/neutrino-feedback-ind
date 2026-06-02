"""Generate pulse/noise sweeps and analyze pile-up with an optimal filter."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from math import ceil, pi, sqrt
import os
from typing import Any, Callable

import numpy as np

from sensor import Sensor


DATASET_FORMAT = "neutrino-feedback-ind-pileup-sweeps-v3"
DEFAULT_MAX_SWEEP_SAMPLES = 262144
MAX_NOISE_BATCH_SPECTRAL_POINTS = 16_000_000
OBSOLETE_DATASET_FORMATS = {
    "neutrino-feedback-ind-pileup-sweeps-v1",
    "neutrino-feedback-ind-pileup-sweeps-v2",
}


def _max_sweep_samples() -> int:
    value = os.environ.get("PILEUP_SWEEP_MAX_SAMPLES", "").strip()
    if not value:
        return DEFAULT_MAX_SWEEP_SAMPLES
    try:
        return max(1024, int(value))
    except ValueError:
        return DEFAULT_MAX_SWEEP_SAMPLES
READOUT_SPECS = {
    "Phase": {"key": "phi", "index": 1, "unit": "rad", "asd_unit": "rad/sqrt(Hz)", "scale": "identity"},
    "Amplitude": {"key": "r", "index": 0, "unit": "1", "asd_unit": "1/sqrt(Hz)", "scale": "identity"},
    "L2": {"key": "L2", "index": 3, "unit": "H", "asd_unit": "H/sqrt(Hz)", "scale": "dL2_dT"},
}


def _readout_spec(readout: str) -> dict[str, str | int]:
    if readout not in READOUT_SPECS:
        raise ValueError(f"unsupported pulse-sweep readout: {readout}")
    return READOUT_SPECS[readout]


def _readout_scale(sensor: Sensor, readout: str) -> float:
    spec = _readout_spec(readout)
    if spec.get("scale") == "dL2_dT":
        return float(sensor.dL2_dT_H_per_K)
    return 1.0


def _select_readout(sensor: Sensor, values: np.ndarray, readout: str) -> np.ndarray:
    spec = _readout_spec(readout)
    return np.asarray(values, dtype=complex)[int(spec["index"])] * _readout_scale(sensor, readout)


def _time_constants_s(sensor: Sensor) -> tuple[float, float]:
    eigs = np.asarray(sensor.mt_eigenvalues, dtype=complex)
    stable = np.real(eigs) < 0.0
    if not np.any(stable):
        raise ValueError("pulse generation requires at least one stable decay mode")
    taus = -1.0 / np.real(eigs[stable])
    return float(np.min(taus)), float(np.max(taus))


def _modal_dt_target_s(sensor: Sensor, tau_fast_s: float) -> float:
    max_oscillation = float(np.max(np.abs(np.imag(np.asarray(sensor.mt_eigenvalues, dtype=complex)))))
    dt_s = tau_fast_s / 24.0
    if max_oscillation > 0.0:
        dt_s = min(dt_s, 2.0 * pi / (6.0 * max_oscillation))
    return max(dt_s, 1.0e-9)


def recommended_time_grid(sensor: Sensor, *, lag_stop_us: float) -> tuple[float, float]:
    """Return eigenmode-based recommended sample time and record duration."""
    if lag_stop_us < 0.0:
        raise ValueError("pulse lag stop must be nonnegative")
    tau_fast_s, tau_decay_s = _time_constants_s(sensor)
    sample_time_s = _modal_dt_target_s(sensor, tau_fast_s)
    lag_stop_s = lag_stop_us * 1.0e-6
    record_duration_s = 10.0 * tau_decay_s + 2.0 * lag_stop_s
    return sample_time_s, record_duration_s


ProgressCallback = Callable[[str], None]


def _report(progress: ProgressCallback | None, message: str) -> None:
    if progress is not None:
        progress(message)


def readout_noise_asd_per_rtHz(
    sensor: Sensor, freqs_hz: np.ndarray, readout: str, progress: ProgressCallback | None = None
) -> np.ndarray:
    """Return total selected-readout ASD including branch and amplifier noise."""
    idx = int(_readout_spec(readout)["index"])
    scale = _readout_scale(sensor, readout)
    fixed_sources = (
        sensor.n_johnson_A_1(),
        sensor.n_johnson_phi_1(),
        sensor.n_johnson_A_2(),
        sensor.n_johnson_phi_2(),
        sensor.n_phonon_1(),
        sensor.n_phonon_2(),
        sensor.n_electronic_A_1(),
        sensor.n_electronic_phi_1(),
        sensor.n_electronic_A_2(),
        sensor.n_electronic_phi_2(),
    )
    freqs = np.asarray(freqs_hz, dtype=float)
    eval_freqs = np.where(freqs > 0.0, freqs, 1.0)
    out = np.zeros(freqs.size, dtype=float)
    output_matrix = sensor.state_output_matrix()
    n_state = output_matrix.shape[1]
    n_internal = len(fixed_sources) + 1
    static_rhs = np.zeros((n_state, n_internal), dtype=complex)
    for source_idx, source in enumerate(fixed_sources):
        static_rhs[:, source_idx] = sensor._state_source_vector(source)

    amplifier_sources = (sensor.n_amplifier_A(), sensor.n_amplifier_phi())
    amplifier_rhs = np.zeros((n_state, len(amplifier_sources)), dtype=complex)
    amplifier_direct = np.zeros((4, len(amplifier_sources)), dtype=complex)
    gain = float(sensor.feedback_heater_gain_W_per_rad)
    derivative_enabled = sensor.feedback_heater_derivative_state_enabled
    factor = float(sensor.feedback_heater_derivative_filter_factor) if derivative_enabled else 0.0
    for source_idx, source in enumerate(amplifier_sources):
        q_r, q_phi = source[0], source[1]
        if derivative_enabled:
            amplifier_rhs[3, source_idx] = gain * (1.0 + factor) * q_phi
            amplifier_rhs[4, source_idx] = q_phi
        else:
            amplifier_rhs[3, source_idx] = gain * q_phi
        amplifier_direct[0, source_idx] = q_r
        amplifier_direct[1, source_idx] = q_phi

    # M(f) is affine in frequency; construct and solve a batch of dense FFT bins at once.
    m0 = sensor.m_matrix_array(0.0)
    dm_df = (sensor.m_matrix_array(1.0) - sensor.m_matrix_array(-1.0)) / 2.0
    chunk_size = 4096
    for start in range(0, freqs.size, chunk_size):
        stop = min(start + chunk_size, freqs.size)
        batch_freqs = eval_freqs[start:stop]
        matrices = m0[None, :, :] + batch_freqs[:, None, None] * dm_df[None, :, :]
        rhs = np.broadcast_to(static_rhs, (stop - start, n_state, n_internal)).copy()
        for batch_idx, f_hz in enumerate(batch_freqs):
            rhs[batch_idx, :, -1] = sensor._state_source_vector(sensor.n_tls_phi_at_hz(float(f_hz)))
        internal_states = np.linalg.solve(matrices, rhs)
        internal_measured = np.einsum("os,bsn->bon", output_matrix, internal_states)
        amp_rhs = np.broadcast_to(amplifier_rhs, (stop - start, n_state, len(amplifier_sources)))
        amplifier_states = np.linalg.solve(matrices, amp_rhs)
        amplifier_measured = np.einsum("os,bsn->bon", output_matrix, amplifier_states) + amplifier_direct[None, :, :]
        out[start:stop] = np.sqrt(
            np.sum(np.abs(internal_measured[:, idx, :]) ** 2, axis=1)
            + np.sum(np.abs(amplifier_measured[:, idx, :]) ** 2, axis=1)
        ) * abs(scale)
        _report(progress, f"{stop} of {freqs.size} noise frequency bins computed")
    return out


def _noise_sweeps(
    asd: np.ndarray,
    n_samples: int,
    dt_s: float,
    count: int,
    rng: np.random.Generator,
    *,
    label: str,
    progress: ProgressCallback | None = None,
) -> np.ndarray:
    traces = np.empty((count, n_samples), dtype=np.float32)
    chunk_size = max(
        1,
        min(
            10,
            int(ceil(count / 20.0)),
            MAX_NOISE_BATCH_SPECTRAL_POINTS // max(1, asd.size),
        ),
    )
    for start in range(0, count, chunk_size):
        stop = min(start + chunk_size, count)
        batch_count = stop - start
        spectra = np.zeros((batch_count, asd.size), dtype=complex)
        if asd.size > 2:
            z = (
                rng.standard_normal((batch_count, asd.size - 2))
                + 1j * rng.standard_normal((batch_count, asd.size - 2))
            ) / np.sqrt(2.0)
            spectra[:, 1:-1] = z * asd[None, 1:-1] * np.sqrt(n_samples / (2.0 * dt_s))
        if n_samples % 2 == 0:
            spectra[:, -1] = rng.standard_normal(batch_count) * asd[-1] * np.sqrt(n_samples / dt_s)
        traces[start:stop] = np.fft.irfft(spectra, n=n_samples, axis=1).astype(np.float32)
        _report(progress, f"{stop} of {count} {label} computed")
    return traces


def estimate_asd_per_rtHz(traces: np.ndarray, dt_s: float) -> np.ndarray:
    yf = np.fft.rfft(np.asarray(traces, dtype=float), axis=1)
    psd = (2.0 * dt_s / traces.shape[1]) * np.mean(np.abs(yf) ** 2, axis=0)
    psd[0] *= 0.5
    if traces.shape[1] % 2 == 0:
        psd[-1] *= 0.5
    return np.sqrt(np.maximum(psd, 0.0))


def _shifted_fft_pulse(
    template_fft_per_eV: np.ndarray,
    freqs_hz: np.ndarray,
    n_samples: int,
    energy_eV: float,
    time_s: float,
) -> np.ndarray:
    shift = np.exp(-2.0j * pi * freqs_hz * float(time_s))
    return np.fft.irfft(float(energy_eV) * template_fft_per_eV * shift, n=n_samples)


def _shifted_truncated_pulse(
    template_per_eV: np.ndarray,
    dt_s: float,
    energy_eV: float,
    time_s: float,
) -> np.ndarray:
    n_samples = int(template_per_eV.size)
    t_eval = np.arange(n_samples, dtype=float) * dt_s - float(time_s)
    return float(energy_eV) * np.interp(
        t_eval,
        np.arange(n_samples, dtype=float) * dt_s,
        np.asarray(template_per_eV, dtype=float),
        left=0.0,
        right=0.0,
    )


def generate_pulse_sweep_dataset(
    sensor: Sensor,
    *,
    readout: str = "Phase",
    n_single: int,
    n_double: int,
    n_noise: int,
    n_energies: int,
    minimum_smaller_energy_eV: float,
    maximum_smaller_energy_eV: float,
    lag_start_us: float,
    lag_stop_us: float,
    sample_time_s: float | None = None,
    record_duration_s: float | None = None,
    random_seed: int | None = None,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Generate noisy single-event and pile-up records for one readout."""
    spec = _readout_spec(readout)
    if min(n_single, n_double, n_noise, n_energies) < 1:
        raise ValueError("sweep and energy counts must be positive")
    if minimum_smaller_energy_eV <= 0.0 or maximum_smaller_energy_eV <= minimum_smaller_energy_eV:
        raise ValueError("smaller-pulse energy bounds must be increasing and positive")
    if lag_start_us <= 0.0 or lag_stop_us < lag_start_us:
        raise ValueError("pulse lag bounds must be positive and increasing")
    ho_energy_eV = float(sensor.ho_decay_energy_eV)
    if maximum_smaller_energy_eV > 0.5 * ho_energy_eV:
        raise ValueError("maximum smaller pulse must be <= half the Ho event energy")

    rng = np.random.default_rng(random_seed)
    tau_fast_s, tau_decay_s = _time_constants_s(sensor)
    lag_stop_s = lag_stop_us * 1.0e-6
    pretrigger_s = tau_decay_s
    default_dt_s, default_duration_s = recommended_time_grid(sensor, lag_stop_us=lag_stop_us)
    dt_s = default_dt_s if sample_time_s is None else float(sample_time_s)
    requested_duration_s = default_duration_s if record_duration_s is None else float(record_duration_s)
    if dt_s <= 0.0:
        raise ValueError("sample time must be positive")
    if requested_duration_s <= 0.0:
        raise ValueError("record duration must be positive")
    n_samples = int(ceil(requested_duration_s / dt_s))
    n_samples = max(n_samples, 16)
    max_samples = _max_sweep_samples()
    if n_samples > max_samples:
        raise ValueError(
            f"requested sweep requires {n_samples} samples; limit is {max_samples}. "
            "Raise PILEUP_SWEEP_MAX_SAMPLES if this dataset size is intentional."
        )
    actual_duration_s = n_samples * dt_s
    minimum_duration_s = lag_stop_s + 3.0 * dt_s
    if actual_duration_s < minimum_duration_s:
        raise ValueError(
            f"record duration must be >= {minimum_duration_s:.6g} s to contain the maximum pulse lag"
        )
    time_s = np.arange(n_samples, dtype=float) * dt_s
    freqs_hz = np.fft.rfftfreq(n_samples, dt_s)
    preferred_reference_time_s = ceil(pretrigger_s / dt_s) * dt_s
    preferred_start_span_s = lag_stop_s
    preferred_timing_fits = (
        preferred_reference_time_s + preferred_start_span_s + lag_stop_s + 2.0 * dt_s
        <= actual_duration_s
    )
    if preferred_timing_fits:
        reference_time_s = preferred_reference_time_s
        start_span_s = preferred_start_span_s
        timing_placement = "preferred"
    else:
        start_span_s = min(lag_stop_s, max(0.0, actual_duration_s - lag_stop_s - 2.0 * dt_s))
        timing_window_s = start_span_s + lag_stop_s
        reference_time_s = 0.5 * (actual_duration_s - timing_window_s)
        reference_time_s = max(dt_s, reference_time_s)
        timing_placement = "centered"
    start_span_clipped = start_span_s < lag_stop_s
    timing_fit_to_record = not preferred_timing_fits or start_span_clipped
    template_truncated = actual_duration_s < default_duration_s

    _report(progress, "Computing event template...")
    response_per_eV = np.real(_select_readout(sensor, sensor.event_response_time_per_eV(time_s), readout))
    _report(progress, "Computing model noise ASD...")
    model_asd = readout_noise_asd_per_rtHz(sensor, freqs_hz, readout, progress=progress)
    noise_records = _noise_sweeps(
        model_asd, n_samples, dt_s, n_noise, rng, label="noise sweeps", progress=progress
    )
    # Fits apply the event time as an FFT phase shift, so the stored template
    # must represent an event beginning at t=0 rather than at the pretrigger.
    template = response_per_eV
    template_fft = np.fft.rfft(template)
    if not np.any(np.abs(template_fft) > 0.0):
        raise ValueError("selected readout has zero pulse response")

    single_times = reference_time_s + rng.uniform(0.0, start_span_s, n_single)
    single_energy = np.full(n_single, ho_energy_eV)
    single = _noise_sweeps(
        model_asd, n_samples, dt_s, n_single, rng, label="single sweep noise records", progress=progress
    )
    report_step = max(1, n_single // 20)
    for i in range(n_single):
        single[i] += _shifted_truncated_pulse(response_per_eV, dt_s, single_energy[i], single_times[i])
        completed = i + 1
        if completed == n_single or completed % report_step == 0:
            _report(progress, f"{completed} of {n_single} single sweeps computed")

    small_grid = np.geomspace(minimum_smaller_energy_eV, maximum_smaller_energy_eV, n_energies)
    smaller = np.resize(small_grid, n_double)
    rng.shuffle(smaller)
    assign_first = rng.random(n_double) < 0.5
    energies = np.column_stack((np.where(assign_first, smaller, ho_energy_eV - smaller), np.where(assign_first, ho_energy_eV - smaller, smaller)))
    first_times = reference_time_s + rng.uniform(0.0, start_span_s, n_double)
    lags_s = np.geomspace(lag_start_us, lag_stop_us, n_double) * 1.0e-6
    rng.shuffle(lags_s)
    double_times = np.column_stack((first_times, first_times + lags_s))
    double = _noise_sweeps(
        model_asd, n_samples, dt_s, n_double, rng, label="pile-up sweep noise records", progress=progress
    )
    report_step = max(1, n_double // 20)
    for i in range(n_double):
        double[i] += _shifted_truncated_pulse(response_per_eV, dt_s, energies[i, 0], double_times[i, 0])
        double[i] += _shifted_truncated_pulse(response_per_eV, dt_s, energies[i, 1], double_times[i, 1])
        completed = i + 1
        if completed == n_double or completed % report_step == 0:
            _report(progress, f"{completed} of {n_double} pile-up sweeps computed")

    _report(progress, "Computing optimal-filter noise estimate...")
    estimated_asd = estimate_asd_per_rtHz(noise_records, dt_s)
    psd = np.maximum(estimated_asd**2, np.finfo(float).tiny)
    weights = np.zeros_like(psd)
    weights[1:] = 1.0 / psd[1:]
    normalization = float(np.fft.irfft(np.abs(template_fft) ** 2 * weights, n=n_samples)[0])
    warnings = []
    if template_truncated:
        warnings.append(
            "Record is shorter than the recommended modal tail duration; "
            "template and optimal filter are truncated to the record length."
        )
    if timing_fit_to_record:
        warnings.append(
            "Pulse timing was centered to fit the requested record length while preserving the full lag range."
        )
    return {
        "format": DATASET_FORMAT,
        "created": datetime.now().isoformat(timespec="seconds"),
        "sensor_inputs": asdict(sensor.inputs),
        "readout": str(spec["key"]),
        "readout_label": readout,
        "units": {"time": "s", "trace": str(spec["unit"]), "energy": "eV", "asd": str(spec["asd_unit"])},
        "generation": {
            "n_samples": n_samples,
            "dt_s": dt_s,
            "requested_record_duration_s": requested_duration_s,
            "record_duration_s": actual_duration_s,
            "recommended_dt_s": default_dt_s,
            "recommended_record_duration_s": default_duration_s,
            "tau_fast_s": tau_fast_s,
            "tau_decay_s": tau_decay_s,
            "minimum_fft_frequency_hz": float(freqs_hz[1]),
            "lag_start_us": lag_start_us,
            "lag_stop_us": lag_stop_us,
            "preferred_reference_time_s": preferred_reference_time_s,
            "reference_time_s": reference_time_s,
            "timing_placement": timing_placement,
            "start_time_span_s": start_span_s,
            "start_time_span_clipped": start_span_clipped,
            "template_duration_s": actual_duration_s,
            "template_truncated": template_truncated,
            "timing_fit_to_record": timing_fit_to_record,
            "warning": " ".join(warnings),
            "search_start_s": reference_time_s,
            "search_stop_s": reference_time_s + start_span_s + lag_stop_s,
        },
        "time_s": time_s,
        "frequencies_hz": freqs_hz,
        "energy_grid_eV": small_grid,
        "noise": {"trace": noise_records, "model_asd_per_rtHz": model_asd, "estimated_asd_per_rtHz": estimated_asd},
        "single": {"trace": single, "energies_eV": single_energy, "total_energy_eV": single_energy.copy(), "pulse_times_s": single_times[:, None], "lag_s": np.zeros(n_single)},
        "double": {"trace": double, "energies_eV": energies, "total_energy_eV": np.sum(energies, axis=1), "pulse_times_s": double_times, "lag_s": lags_s},
        "optimal_filter": {"template_fft_per_eV": template_fft, "weights": weights, "noise_psd_per_Hz": psd, "normalization": normalization},
    }


def _continuous_correlation_score(
    cross_spectrum: np.ndarray, freqs_hz: np.ndarray, n_samples: int, time_s: float
) -> float:
    phase = np.exp(2.0j * pi * freqs_hz * float(time_s))
    score = float(np.real(cross_spectrum[0]))
    if n_samples % 2 == 0:
        score += float(np.real(cross_spectrum[-1] * phase[-1]))
        interior = slice(1, -1)
    else:
        interior = slice(1, None)
    score += 2.0 * float(np.real(np.dot(cross_spectrum[interior], phase[interior])))
    return score / n_samples


def _maximize_time_score(score: Callable[[float], float], left_s: float, right_s: float, tolerance_s: float) -> float:
    ratio = (sqrt(5.0) - 1.0) / 2.0
    c_s = right_s - ratio * (right_s - left_s)
    d_s = left_s + ratio * (right_s - left_s)
    c_score = score(c_s)
    d_score = score(d_s)
    while right_s - left_s > tolerance_s:
        if c_score > d_score:
            right_s, d_s, d_score = d_s, c_s, c_score
            c_s = right_s - ratio * (right_s - left_s)
            c_score = score(c_s)
        else:
            left_s, c_s, c_score = c_s, d_s, d_score
            d_s = left_s + ratio * (right_s - left_s)
            d_score = score(d_s)
    return 0.5 * (left_s + right_s)


def _refine_time_score(
    cross_spectrum: np.ndarray,
    freqs_hz: np.ndarray,
    n_samples: int,
    initial_time_s: float,
    left_s: float,
    right_s: float,
    tolerance_s: float,
) -> float:
    omega = 2.0 * pi * freqs_hz
    multiplicity = np.full(cross_spectrum.size, 2.0)
    multiplicity[0] = 1.0
    if n_samples % 2 == 0:
        multiplicity[-1] = 1.0
    weighted = multiplicity * cross_spectrum
    time_s = initial_time_s
    for _ in range(10):
        phase = np.exp(1.0j * omega * time_s)
        first = float(np.real(np.dot(weighted * (1.0j * omega), phase)) / n_samples)
        second = float(np.real(np.dot(weighted * (-omega**2), phase)) / n_samples)
        if not np.isfinite(second) or second >= 0.0:
            break
        candidate_s = float(np.clip(time_s - first / second, left_s, right_s))
        if abs(candidate_s - time_s) <= tolerance_s:
            return candidate_s
        time_s = candidate_s
    return _maximize_time_score(
        lambda value_s: _continuous_correlation_score(cross_spectrum, freqs_hz, n_samples, value_s),
        left_s,
        right_s,
        tolerance_s,
    )


def _zero_lag_irfft_value(spectrum: np.ndarray, n_samples: int) -> float:
    if spectrum.size == 0:
        return 0.0
    total = float(np.real(spectrum[0]))
    if n_samples % 2 == 0 and spectrum.size > 1:
        total += float(np.real(spectrum[-1]))
        interior = slice(1, -1)
    else:
        interior = slice(1, None)
    if spectrum[interior].size:
        total += 2.0 * float(np.sum(np.real(spectrum[interior])))
    return total / n_samples


def _fit_one_trace(
    dataset: dict[str, Any],
    trace: np.ndarray,
    coarse_weights: np.ndarray | None = None,
) -> tuple[float, float, float]:
    n_samples = int(dataset["generation"]["n_samples"])
    dt_s = float(dataset["generation"]["dt_s"])
    freqs_hz = np.asarray(dataset["frequencies_hz"], dtype=float)
    template_fft = np.asarray(dataset["optimal_filter"]["template_fft_per_eV"], dtype=complex)
    weights = np.asarray(dataset["optimal_filter"]["weights"], dtype=float)
    yf = np.fft.rfft(np.asarray(trace, dtype=float))
    cross_spectrum = np.conjugate(template_fft) * yf * weights
    if coarse_weights is None:
        # Do not use the top FFT bins for coarse triggering. Underdamped pulses can
        # otherwise select an aliased high-frequency lobe before sub-sample fitting.
        coarse_weights = weights.copy()
        coarse_weights[freqs_hz > 0.9 * freqs_hz[-1]] = 0.0
    corr = np.fft.irfft(np.conjugate(template_fft) * yf * coarse_weights, n=n_samples)
    i0 = max(1, int(dataset["generation"]["search_start_s"] / dt_s))
    i1 = min(n_samples - 2, int(dataset["generation"]["search_stop_s"] / dt_s) + 1)
    peak = i0 + int(np.argmax(corr[i0:i1 + 1]))
    coarse_time_s = peak * dt_s
    left_s = max(float(dataset["generation"]["search_start_s"]), coarse_time_s - 0.6 * dt_s)
    right_s = min(float(dataset["generation"]["search_stop_s"]), coarse_time_s + 0.6 * dt_s)
    time_fit = _refine_time_score(
        cross_spectrum,
        freqs_hz,
        n_samples,
        coarse_time_s,
        left_s,
        right_s,
        dt_s * 1.0e-4,
    )
    shift = np.exp(-2.0j * pi * freqs_hz * time_fit)
    shifted = template_fft * shift
    energy_numerator = _zero_lag_irfft_value(np.conjugate(shifted) * yf * weights, n_samples)
    energy_fit = float(energy_numerator / dataset["optimal_filter"]["normalization"])
    residual = yf - energy_fit * shifted
    valid = slice(1, -1)
    valid_size = max(residual.size - 2, 0)
    chi2 = float(
        (4.0 * dt_s / n_samples)
        * np.sum(np.abs(residual[valid]) ** 2 / dataset["optimal_filter"]["noise_psd_per_Hz"][valid])
    )
    return energy_fit, time_fit, chi2 / max(2 * valid_size - 2, 1)


def analyze_pulse_sweep_dataset(
    dataset: dict[str, Any],
    progress: ProgressCallback | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    if dataset.get("format") in OBSOLETE_DATASET_FORMATS:
        raise ValueError("dataset uses an obsolete pulse timing format; regenerate the pulse sweeps")
    if dataset.get("format") != DATASET_FORMAT:
        raise ValueError("not a supported pile-up sweep dataset")
    out: dict[str, dict[str, np.ndarray]] = {}
    weights = np.asarray(dataset["optimal_filter"]["weights"], dtype=float)
    freqs_hz = np.asarray(dataset["frequencies_hz"], dtype=float)
    coarse_weights = weights.copy()
    coarse_weights[freqs_hz > 0.9 * freqs_hz[-1]] = 0.0
    for category in ("single", "double"):
        traces = dataset[category]["trace"]
        fits = np.empty((len(traces), 3), dtype=float)
        report_step = max(1, len(traces) // 50)
        for idx, trace in enumerate(traces):
            fits[idx] = _fit_one_trace(dataset, trace, coarse_weights)
            completed = idx + 1
            if completed == len(traces) or completed % report_step == 0:
                _report(progress, f"{completed} of {len(traces)} {category} traces analyzed")
        out[category] = {
            "true_total_energy_eV": np.asarray(dataset[category]["total_energy_eV"]),
            "lag_s": np.asarray(dataset[category]["lag_s"]),
            "fit_energy_eV": fits[:, 0],
            "fit_time_s": fits[:, 1],
            "reduced_chi2": fits[:, 2],
        }
    return out
