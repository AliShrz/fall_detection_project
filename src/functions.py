import os
import zipfile
import numpy as np
import pandas as pd


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




