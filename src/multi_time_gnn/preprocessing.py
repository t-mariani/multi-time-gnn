import mne
from scipy.signal import butter, filtfilt, resample

def highpass_filter(data, fs, cutoff):
    """Apply a high-pass Butterworth
    Args:
        data: np.ndarray, shape (..., time)
        fs: float, sampling frequency in Hz
        cutoff: float, cutoff frequency in Hz
    """
    nyquist = fs / 2
    normal_cutoff = cutoff / nyquist
    b, a = butter(4, normal_cutoff, btype='high', analog=False) # 4th order Butterworth
    filtered_data = filtfilt(b, a, data, axis=-1)
    return filtered_data

def resample_data(data, original_fs, target_fs):
    """Resample data from original_fs to target_fs."""
    num_samples = int(data.shape[-1] * target_fs / original_fs)
    resampled_data = resample(data, num_samples, axis=-1)
    return resampled_data


def preprocess_eeg(data, config):
    """
    Preprocess EEG data with high-pass filtering and resampling.
    
    Args:
        data: EEG signal array (channels, samples) or (samples,)
        config: configuration object with attributes:
            - eeg_highpass_cutoff: cutoff frequency for high-pass filter
            - eeg_target_fs: target sampling frequency
            - path_eeg: path to EEG data to get sampling freq
    """
    sampling_freq =  mne.io.read_raw_bdf(config.path_eeg, preload=False).info["sfreq"]
    filtered_data = highpass_filter(data, sampling_freq, config.eeg_highpass_cutoff)
    resampled_data = resample_data(filtered_data, sampling_freq, config.eeg_target_fs)
    return resampled_data