import os
import zipfile
import numpy as np
import pandas as pd

from scipy.stats import skew, kurtosis, entropy
from scipy.signal import welch

def read_zip(zip_path: str, base_path_in_zip: str, subject_ids: list):
    """
    Read SisFall sensor data from a ZIP archive.

    Parameters
    ----------
    zip_path         : full path to the ZIP file
    base_path_in_zip : folder name inside the ZIP (e.g. "SisFall_dataset")
    subject_ids      : list of subject folder names (e.g. ["SA01", "SE14"])

    Returns
    -------
    all_data         : list of np.ndarray, each shape (6, N)
    all_labels       : list of str, "ADL" or "Fall"
    activity_codes   : list of str, e.g. "D01", "F03"
    file_names       : list of str, e.g. "D01_01.txt"
    """
    all_data       = []
    all_labels     = []
    activity_codes = []
    file_names     = []
    counter        = 0
    adl_count      = 0
    fall_count     = 0

    with zipfile.ZipFile(zip_path, "r") as zf:
        for subject_id in subject_ids:
            folder = os.path.join(base_path_in_zip, subject_id)

            for item in zf.namelist():
                if item.startswith(folder + "/") and item.endswith(".txt"):
                    fname = os.path.basename(item)
                    activity_code = fname.split("_")[0]

                    try:
                        with zf.open(item) as f:
                            df = pd.read_csv(
                                f,
                                header=None,
                                delimiter=",",
                                usecols=[0, 1, 2, 3, 4, 5],
                                on_bad_lines="skip",
                            )
                        # shape becomes (6, N) — 6 axes, N time samples
                        data = df.to_numpy().T

                        # determine label from filename
                        if activity_code.startswith("D"):
                            label = "ADL"
                            adl_count += 1
                        elif activity_code.startswith("F"):
                            label = "Fall"
                            fall_count += 1
                        else:
                            label = "Unknown"

                        all_data.append(data)
                        all_labels.append(label)
                        activity_codes.append(activity_code)
                        file_names.append(fname)
                        counter += 1

                        if counter % 200 == 0:
                            print(f"Progress: {counter} files read...")

                    except Exception as e:
                        print(f"Error reading {item}: {e}")

    print(f"\nDone")
    print(f"Total files : {counter}")
    print(f"ADL         : {adl_count}")
    print(f"Fall        : {fall_count}")

    return all_data, all_labels, activity_codes, file_names

##################################################

# find position of max valur in signal, then keep a desired size from both sides
def keep_from_peak(data_list, window_size):
    """
    For each signal in data_list, keep a slice of length 2*window_size
    centered around the peak point (maximum combined absolute value of first 3 axes).
    """
    all_cleaned_data = []
    
    for i, signal in enumerate(data_list):
        # Compute the combined absolute magnitude of the first 3 axes
        combined = np.abs(signal[0]) + np.abs(signal[1]) + np.abs(signal[2])
        
        # Find the index of the peak
        peak_index = np.argmax(combined)
        
        # Determine start and end of window
        start = max(0, peak_index - window_size)
        end = min(signal.shape[1], peak_index + window_size)

        # Slice the signal and store it
        cleaned = signal[:, start:end]
        all_cleaned_data.append(cleaned)
        
        #print(f"Signal {i+1}: Original shape = {signal.shape}, Kept shape = {cleaned.shape}")
    # print(f"{len(all_cleaned_data)} files processed,")
    min_length1 = min(signal.shape[1] for signal in all_cleaned_data)
    max_length1 = max(signal.shape[1] for signal in all_cleaned_data)
    print(f"{len(all_cleaned_data)} files processed \nMin length: {min_length1} \nMax length: {max_length1}")
    return all_cleaned_data

##################################################

def length_fix(data_list, length):
    fixed_list = []
    for i, signal in enumerate(data_list):
        data_length = signal.shape[1]
        if data_length < length:
            pad_width = length - data_length
            left_pad = pad_width // 2
            right_pad = pad_width - left_pad

            # Get edge values
            left_vals = signal[:, 0:1]
            right_vals = signal[:, -1:]

            # Pad using edge values instead of zeros
            left_padding = np.repeat(left_vals, left_pad, axis=1)
            right_padding = np.repeat(right_vals, right_pad, axis=1)
            signal = np.concatenate([left_padding, signal, right_padding], axis=1)
        else:
            signal = signal[:, :length]

        fixed_list.append(signal)

    min_length = min(signal.shape[1] for signal in fixed_list)
    max_length = max(signal.shape[1] for signal in fixed_list)
    print(f"{len(fixed_list)} files processed, Min length: {min_length}, Max length: {max_length}")
    return fixed_list

##################################################

def split_and_add(input_data, length, file_names):
    """
    Splits signals and creates file/activity lists (optimized).
    """
    new_data = []
    new_file_names = []
    activity_code = file_names[0].split('_')[0]

    for i in range(len(input_data)):
        file_name = file_names[i]
        signal = input_data[i]
        signl_length = signal.shape[1]
        n = signl_length // length
        start = 0
        end = length
        for _ in range(n):
            trimmed = signal[:, start:end]
            new_data.append(trimmed)
            new_file_names.append(file_name)

    new_activity_code_list = [activity_code] * len(new_data)  # Create new activity code list
    new_data = np.array(new_data)  # Convert to numpy array
    print(f"number of new {activity_code} data: {len(new_data)} with shape: {np.shape(new_data)}, "
          f"file_names :{len(new_file_names)}, activities: {len(new_activity_code_list)} ")

    return new_data, new_file_names, new_activity_code_list

##################################################

def group_by_activity(X, activity_codes, code):
    """Return list of signals matching a specific activity code."""
    return [X[i] for i in range(len(X)) if activity_codes[i] == code]

def group_by_prefix(X, activity_codes, prefix):
    """Return list of signals whose code starts with a prefix (e.g. 'F' for all falls)."""
    return [X[i] for i in range(len(X)) if activity_codes[i].startswith(prefix)]

##################################################

# remove parts of signal that have no movement
def idle_remover(data_list, window_size, scale, mode):    # modes: 'acc', 'gyro', 'both'

    all_cleaned_data = []

    for i in range(len(data_list)):
        signal = data_list[i]  # shape: (6, N)
        acc = signal[0:3]
        gyro = signal[3:6]

        # Precompute thresholds
        acc_combined = np.abs(acc[0]) + np.abs(acc[1]) + np.abs(acc[2])
        gyro_combined = np.abs(gyro[0]) + np.abs(gyro[1]) + np.abs(gyro[2])

        acc_threshold = np.var(acc_combined) / scale
        gyro_threshold = np.var(gyro_combined) / scale

        windowed_data = []

        for j in range(0, signal.shape[1] - window_size + 1, window_size):
            acc_window = acc_combined[j:j + window_size]
            gyro_window = gyro_combined[j:j + window_size]

            acc_var = np.var(acc_window)
            gyro_var = np.var(gyro_window)

            # Check condition based on selected mode
            keep = False
            if mode == 'acc':
                keep = acc_var >= acc_threshold
            elif mode == 'gyro':
                keep = gyro_var >= gyro_threshold
            elif mode == 'both':
                keep = (acc_var >= acc_threshold) and (gyro_var >= gyro_threshold)
            else:
                raise ValueError("Invalid mode. Use 'acc', 'gyro', or 'both'.")

            if keep:
                windowed_data.append(signal[:, j:j + window_size])

        if windowed_data:
            cleaned_signal = np.concatenate(windowed_data, axis=1)
        else:
            cleaned_signal = np.empty((6, 0))

        # print(f"Signal {i+1}: Original shape = {signal.shape}, Cleaned shape = {cleaned_signal.shape}")
        all_cleaned_data.append(cleaned_signal)

    min_length = min(signal.shape[1] for signal in all_cleaned_data)
    max_length = max(signal.shape[1] for signal in all_cleaned_data)
    print(f"{len(all_cleaned_data)} files processed, Min length: {min_length}, Max length: {max_length}")
    return all_cleaned_data

##################################################

def split_and_center(data_list, length):
    list_1 = []
    list_2 = []
    start_1 = end_1 = start_2 = end_2 = None
    for i, signal in enumerate(data_list):
        data_length = signal.shape[1]  # Use .shape[1] instead of len(signal)
        # signal = data_list[i]
        # data_length = len(signal)
        if data_length > length * 2:
            start_1 = (data_length // 2 - length) // 2
            end_1 = start_1 + length
            start_2 = end_1 + (start_1 * 2)
            end_2 = start_2 + length
        else:
            start_1 = 0
            end_1 = length
            end_2 = data_length
            start_2 = end_2 - length
            # print(f"length of {i} is smaller than {length}")
        
        cleaned_signal_1 = signal[:, start_1:end_1] if start_1 is not None and end_1 is not None else np.empty((6, 0))
        cleaned_signal_2 = signal[:, start_2:end_2] if start_2 is not None and end_2 is not None else np.empty((6, 0))
        list_1.append(cleaned_signal_1)
        list_2.append(cleaned_signal_2)

    min_length1 = min(signal.shape[1] for signal in list_1)
    max_length1 = max(signal.shape[1] for signal in list_1)
    min_length2 = min(signal.shape[1] for signal in list_2)
    max_length2 = max(signal.shape[1] for signal in list_2)
    print(f"{len(list_1)} files processed, Min length: {min_length1}, Max length: {max_length1}"
          f"\n{len(list_2)} files processed, Min length: {min_length2}, Max length: {max_length2}")
    return list_1, list_2

##################################################

def standard_deviation_magnitude(signal_list, window_size=50, step_size=25):
    """
    Compute Standard Deviation Magnitude feature from a list of 6-row sensor data arrays.
    
    Parameters:
        signal_list: list of np.ndarray, each of shape (6, n_samples)
                        Rows 0-2: Accelerometer [x, y, z]
                        Rows 3-5: Gyroscope [x, y, z] (ignored)
        window_size: int, number of samples in each window
        step_size: int, sliding step size

    Returns:
        List of C9 values (one per signal input)
    """
    features = []

    for data in signal_list:
        acc_x = data[0, :]
        acc_y = data[1, :]
        acc_z = data[2, :]

        values = []

        for i in range(0, acc_x.shape[0] - window_size + 1, step_size):
            seg_x = acc_x[i:i + window_size]
            seg_y = acc_y[i:i + window_size]
            seg_z = acc_z[i:i + window_size]

            std_x = np.std(seg_x)
            std_y = np.std(seg_y)
            std_z = np.std(seg_z)

            value = np.sqrt(std_x**2 + std_y**2 + std_z**2)
            values.append(value)

        # Optionally summarize: mean or max or keep the full list
        # features.append(np.mean(values))  # or np.max(values)
        features.append(values)

    print(f"Number of features extracted (f4): {len(features)}")
    return features

##################################################

def extract_from_high_amp_segments(
    data_list, window_size, step_size, amp_scale, min_gap, before, after, sensor_type='acc'
):
    """
    For each signal in data_list, extract up to two segments where the peak-to-peak amplitude
    is above a threshold (relative to max). Segments are separated by at least `min_gap`.

    Parameters:
        data_list: list of np.ndarray, each of shape (6, n_samples)
        window_size: int, size of each segment to extract
        step_size: int, step size for sliding window
        amp_scale: float, threshold as a fraction of max peak-to-peak value
        min_gap: int, minimum number of samples between two selected segments
        before: int, number of samples before peak to include
        after: int, total number of samples in the segment
        sensor_type: str, 'acc' or 'gyro' to select which sensor to use for peak detection

    Returns:
        list_1: list of first high-amplitude segments
        list_2: list of second high-amplitude segments (or empty if only one segment found)
    """
    list_1 = []
    list_2 = []

    # Choose axes based on sensor_type
    if sensor_type == 'acc':
        selected_data = [signal[0:3, :] for signal in data_list]  # Rows 0–2: Accelerometer
    elif sensor_type == 'gyro':
        selected_data = [signal[3:6, :] for signal in data_list]  # Rows 3–5: Gyroscope
    else:
        raise ValueError("sensor_type must be 'acc' or 'gyro'")

    # Compute peak-to-peak amplitude features
    # selected_features = max_peak_to_peak_amp(selected_data, window_size, step_size)
    # compute Compute Standard Deviation Magnitude
    selected_features = standard_deviation_magnitude(selected_data, window_size, step_size)

    for signal_idx, signal in enumerate(data_list):
        features = selected_features[signal_idx]
        max_val = max(features)
        threshold = amp_scale * max_val

        selected_segments = []
        last_added_index = -min_gap  # so first one can always be added

        for i, value in enumerate(features):
            if value >= threshold:
                segment_start = i * step_size - before
                segment_end = segment_start + after

                if segment_start < 0 :
                    continue  # skip invalid windows
                if segment_end > signal.shape[1]:
                    segment_end = signal.shape[1]

                # Avoid overlap by checking distance to last segment
                if segment_start - last_added_index >= min_gap:
                    selected_segments.append(signal[:, segment_start:segment_end])
                    last_added_index = segment_start

            if len(selected_segments) == 2:
                break

        # Append segments or empty arrays if not enough were found
        if len(selected_segments) >= 1:
            list_1.append(selected_segments[0])
        else:
            list_1.append(np.empty((6, 0)))

        if len(selected_segments) >= 2:
            list_2.append(selected_segments[1])
        else:
            list_2.append(np.empty((6, 0)))

    lengths1 = [s.shape[1] for s in list_1]
    lengths2 = [s.shape[1] for s in list_2]
    print(f"{len(list_1)} files processed using scale: {amp_scale}"
          f" \nMin1: {min(lengths1)}, Max1: {max(lengths1)}"
          f" \nMin2: {min(lengths2)}, Max2: {max(lengths2)}")

    return list_1, list_2

##################################################

def preprocess_pipeline(X, activity_codes, file_names, cfg):
    """
    Apply full preprocessing pipeline to raw signals.

    Parameters
    ----------
    X              : np.array (N,) object array of (6, T) signals
    activity_codes : np.array (N,) activity code per signal
    file_names     : np.array (N,) file name per signal
    cfg            : dict loaded from config.yaml

    Returns
    -------
    X_processed  : np.ndarray (N, 6, 800)
    y_binary     : np.ndarray (N,) — "ADL" or "Fall"
    y_multiclass : np.ndarray (N,) — activity group name
    """
    params        = cfg["preprocessing"]["activity_params"]
    groups        = cfg["preprocessing"]["activity_groups"]
    excluded      = cfg["preprocessing"]["excluded_activities"]
    target_length = cfg["data"]["signal_length"]

    all_signals    = []
    all_binary     = []
    all_multiclass = []

    # ── Falls ────────────────────────────────────────────────────────────
    fall_signals_raw = group_by_prefix(list(X), activity_codes, "F")
    p_fall           = params["F"]
    fall_peaked      = keep_from_peak(fall_signals_raw, p_fall["window_size"])
    fall_fixed       = length_fix(fall_peaked, p_fall["target_length"])
    all_signals.extend(fall_fixed)
    all_binary.extend(["Fall"] * len(fall_fixed))
    all_multiclass.extend(["fall"] * len(fall_fixed))
    print(f"Falls processed: {len(fall_fixed)}")

    # ── ADL activities ────────────────────────────────────────────────────
    unique_codes = np.unique(activity_codes)

    for code in unique_codes:

        # skip excluded activities
        if code in excluded:
            continue

        # skip falls — handled above
        if code.startswith("F"):
            continue

        # skip activities not in params
        if code not in params:
            continue

        # get signals and file names for this activity
        signals = group_by_activity(list(X), activity_codes, code)
        names   = group_by_activity(list(file_names), activity_codes, code)
        p       = params[code]
        method  = p["method"]
        group   = groups[code]

        # ── method: split ─────────────────────────────────────────────
        if method == "split":
            new_data, _, _ = split_and_add(signals, p["window_size"], names)
            all_signals.extend(list(new_data))
            all_binary.extend(["ADL"] * len(new_data))
            all_multiclass.extend([group] * len(new_data))
            print(f"{code} split → {len(new_data)} windows")

        # ── method: keep_from_peak ────────────────────────────────────
        elif method == "keep_from_peak":
            peaked = keep_from_peak(signals, p["window_size"])
            fixed  = length_fix(peaked, p["target_length"])
            all_signals.extend(fixed)
            all_binary.extend(["ADL"] * len(fixed))
            all_multiclass.extend([group] * len(fixed))
            print(f"{code} keep_from_peak → {len(fixed)} signals")

        # ── method: idle_remover_split ────────────────────────────────
        elif method == "idle_remover_split":
            cleaned    = idle_remover(signals, p["window_size"], p["scale"], p["mode"])
            seg1, seg2 = split_and_center(cleaned, target_length)
            fixed1     = length_fix(seg1, target_length)
            fixed2     = length_fix(seg2, target_length)

            if isinstance(group, dict):
                group_seg1 = group["seg1"]
                group_seg2 = group["seg2"]
            else:
                group_seg1 = group
                group_seg2 = group

            all_signals.extend(fixed1)
            all_signals.extend(fixed2)
            all_binary.extend(["ADL"] * len(fixed1))
            all_binary.extend(["ADL"] * len(fixed2))
            all_multiclass.extend([group_seg1] * len(fixed1))
            all_multiclass.extend([group_seg2] * len(fixed2))
            print(f"{code} idle+split → seg1:{len(fixed1)} seg2:{len(fixed2)}")

        # ── method: high_amp ──────────────────────────────────────────
        elif method == "high_amp":
            seg1, seg2 = extract_from_high_amp_segments(
                signals,
                p["window_size"],
                p["step_size"],
                p["amp_scale"],
                p["min_gap"],
                p["before"],
                p["after"],
                p["sensor_type"]
            )
            fixed1 = length_fix(seg1, target_length)
            fixed2 = length_fix(seg2, target_length)

            if isinstance(group, dict):
                group_seg1 = group["seg1"]
                group_seg2 = group["seg2"]
            else:
                group_seg1 = group
                group_seg2 = group

            all_signals.extend(fixed1)
            all_signals.extend(fixed2)
            all_binary.extend(["ADL"] * len(fixed1))
            all_binary.extend(["ADL"] * len(fixed2))
            all_multiclass.extend([group_seg1] * len(fixed1))
            all_multiclass.extend([group_seg2] * len(fixed2))
            print(f"{code} high_amp → seg1:{len(fixed1)} seg2:{len(fixed2)}")

    # ── Combine everything ─────────────────────────────────────────────
    X_processed  = np.array(all_signals)
    y_binary     = np.array(all_binary)
    y_multiclass = np.array(all_multiclass)

    print(f"\nPipeline complete ✅")
    print(f"X_processed  : {X_processed.shape}")
    print(f"Binary       : {dict(zip(*np.unique(y_binary, return_counts=True)))}")
    print(f"Multiclass   : {dict(zip(*np.unique(y_multiclass, return_counts=True)))}")

    return X_processed, y_binary, y_multiclass


def extract_features(signal, fs=200):
    """
    Extract 68 features from a 6-axis IMU signal.

    Parameters
    ----------
    signal : np.ndarray shape (6, N)
             rows 0-2 → accelerometer X, Y, Z
             rows 3-5 → gyroscope X, Y, Z
    fs     : sampling frequency in Hz

    Returns
    -------
    list of 68 floats
    """
    # ── Accelerometer axes ────────────────────────────
    ax, ay, az = signal[0], signal[1], signal[2]

    # ── Gyroscope axes ────────────────────────────────
    gx, gy, gz = signal[3], signal[4], signal[5]

    N = signal.shape[1]
    t = np.arange(N) / fs

    # ── Accelerometer magnitude ───────────────────────
    acc_mag = np.sqrt(ax**2 + ay**2 + az**2)

    # ── Gyroscope magnitude ───────────────────────────
    gyro_mag = np.sqrt(gx**2 + gy**2 + gz**2)

    # ══════════════════════════════════════════════════
    # ACCELEROMETER FEATURES (C1–C33)
    # ══════════════════════════════════════════════════

    # paste your original 33 features here
    # but fix axis ordering: ax=signal[0], ay=signal[1], az=signal[2]
        # Time-domain features
    C1 = np.sqrt(np.mean(acc_mag**2))  # RMS magnitude
    C2 = np.sqrt(np.mean(ax**2 + az**2))  # RMS horizontal (X,Z)
    a_peak_to_peak = np.max(acc_mag) - np.min(acc_mag)
    C3 = np.sqrt(a_peak_to_peak)
    C4 = np.mean(np.arctan2(np.sqrt(ax**2 + az**2), -ay))
    C5 = np.std(np.arctan(np.sqrt(np.mean(ax**2 + az**2)) / np.mean(ay)))
    C6 = np.mean(ax[:-1]) * np.mean(ax[1:]) if N > 1 else 0
    jerk = np.diff(ax) / np.diff(t) if N > 1 else np.array([0])
    C7 = np.mean(np.abs(jerk)) if len(jerk) > 0 else 0

    C8 = np.sqrt(np.std(ax)**2 + np.std(az)**2)

    C9 = np.sqrt(np.std(ax)**2 + np.std(ay)**2 + np.std(az)**2)

    C10 = (np.sum(np.abs(ax)) + np.sum(np.abs(ay)) + np.sum(np.abs(az))) / N
    C11 = (np.sum(np.abs(ax)) + np.sum(np.abs(az))) / N
    C12 = np.sum(acc_mag)
    C13 = np.sum(np.sqrt(ax**2 + az**2))
    vel_x = np.cumsum(ax) / fs
    vel_z = np.cumsum(az) / fs
    C14 = np.sqrt((np.sum(vel_x))**2 + (np.sum(vel_z))**2) / N
    C15, C16, C17 = np.mean(ax), np.mean(ay), np.mean(az)
    C18, C19, C20 = np.std(ax), np.std(ay), np.std(az)
    C21, C22, C23 = skew(ax), skew(ay), skew(az)
    C24, C25, C26 = kurtosis(ax), kurtosis(ay), kurtosis(az)
    C27 = np.sum(np.diff(np.sign(acc_mag - np.mean(acc_mag))) != 0)
    
    # Frequency-domain features
    f, Pxx = welch(acc_mag, fs=fs, nperseg=min(256, N))
    Pxx_norm = Pxx / np.sum(Pxx) if np.sum(Pxx) > 0 else Pxx
    C28 = entropy(Pxx_norm)
    C29 = f[np.argmax(Pxx)] if len(f) > 0 else 0
    C30 = np.sum(Pxx[(f >= 0) & (f <= 5)]) if len(f) > 0 else 0

    # Correlation features
    C31 = np.corrcoef(ax, ay)[0, 1] if N > 1 else 0
    C32 = np.corrcoef(ay, az)[0, 1] if N > 1 else 0
    C33 = np.corrcoef(ax, az)[0, 1] if N > 1 else 0

    # ══════════════════════════════════════════════════
    # GYROSCOPE FEATURES (G1–G29)
    # ══════════════════════════════════════════════════

    # G1  — RMS magnitude
    G1 = np.sqrt(np.mean(gyro_mag**2))  # RMS magnitude

    # G2  — RMS horizontal (X,Z)
    G2 = np.sqrt(np.mean(gx**2 + gz**2))  # RMS horizontal (X,Z)

    # G3  — peak-to-peak sqrt
    g_peak_to_peak = np.max(gyro_mag) - np.min(gyro_mag)
    G3 = np.sqrt(g_peak_to_peak)

    # G6  — autocorrelation X
    G6 = np.mean(gx[:-1]) * np.mean(gx[1:]) if N > 1 else 0

    # G8  — std magnitude horizontal
    G8 = np.sqrt(np.std(gx)**2 + np.std(gz)**2)

    # G9  — std magnitude 3D
    G9 = np.sqrt(np.std(gx)**2 + np.std(gy)**2 + np.std(gz)**2)

    # G10 — mean absolute combined / N
    G10 = (np.sum(np.abs(gx)) + np.sum(np.abs(gy)) + np.sum(np.abs(gz))) / N

    # G11 — mean absolute horizontal / N
    G11 = (np.sum(np.abs(gx)) + np.sum(np.abs(gz))) / N

    # G12 — sum magnitude
    G12 = np.sum(gyro_mag)

    # G13 — sum horizontal magnitude
    G13 = np.sum(np.sqrt(gx**2 + gz**2))

    # G15 — mean gx, G16 — mean gy, G17 — mean gz
    G15, G16, G17 = np.mean(gx), np.mean(gy), np.mean(gz)

    # G18 — std gx, # G19 — std gy, # G20 — std gz
    G18, G19, G20 = np.std(gx), np.std(gy), np.std(gz)
    
    # G21 — skewness gx, # G22 — skewness gy, # G23 — skewness gz
    G21, G22, G23 = skew(gx), skew(gy), skew(gz)

    # G24 — kurtosis gx, # G25 — kurtosis gy, # G26 — kurtosis gz
    G24, G25, G26 = kurtosis(gx), kurtosis(gy), kurtosis(gz)

    # G27 — zero crossing rate of gyro magnitude
    G27 = np.sum(np.diff(np.sign(gyro_mag - np.mean(gyro_mag))) != 0)

    # Frequency-domain features
    f, Pxx_g = welch(gyro_mag, fs=fs, nperseg=min(256, N))
    Pxx_g_norm = Pxx_g / np.sum(Pxx_g) if np.sum(Pxx_g) > 0 else Pxx_g

    # G28 — spectral entropy of gyro magnitude
    G28 = entropy(Pxx_g_norm)

    # G29 — dominant frequency of gyro magnitude
    G29 = f[np.argmax(Pxx_g)] if len(f) > 0 else 0

    # G30 — low frequency energy of gyro magnitude
    G30 = np.sum(Pxx_g[(f >= 0) & (f <= 5)]) if len(f) > 0 else 0

    # Correlation features
    # G31 — corr(gx, gy)
    # G32 — corr(gy, gz)
    # G33 — corr(gx, gz)
    G31 = np.corrcoef(gx, gy)[0, 1] if N > 1 else 0
    G32 = np.corrcoef(gy, gz)[0, 1] if N > 1 else 0
    G33 = np.corrcoef(gx, gz)[0, 1] if N > 1 else 0


    # ══════════════════════════════════════════════════
    # CROSS-SENSOR FEATURES (X1–X6)
    # ══════════════════════════════════════════════════

    # X1 — corr(acc_x, gyro_x)
    X1 = np.corrcoef(ax, gx)[0, 1] if N > 1 else 0

    # X2 — corr(acc_y, gyro_y)
    X2 = np.corrcoef(ay, gy)[0, 1] if N > 1 else 0

    # X3 — corr(acc_z, gyro_z)
    X3 = np.corrcoef(az, gz)[0, 1] if N > 1 else 0

    # X4 — corr(acc_mag, gyro_mag)
    X4 = np.corrcoef(acc_mag, gyro_mag)[0, 1] if N > 1 else 0

    # X5 — corr(acc_x, gyro_z)
    X5 = np.corrcoef(ax, gz)[0, 1] if N > 1 else 0

    # X6 — corr(acc_z, gyro_x)
    X6 = np.corrcoef(az, gx)[0, 1] if N > 1 else 0

    return [
        C1, C2, C3, C4, C5, C6, C7, C8, C9, C10,
        C11, C12, C13, C14, C15, C16, C17, C18, C19, C20,
        C21, C22, C23, C24, C25, C26, C27, C28, C29, C30,
        C31, C32, C33,
        G1, G2, G3, G6, G8, G9, G10, G11, G12, G13,
        G15, G16, G17, G18, G19, G20,
        G21, G22, G23, G24, G25, G26, G27, G28, G29, G30,
        G31, G32, G33,
        X1, X2, X3, X4, X5, X6,
    ]










