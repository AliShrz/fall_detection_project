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

########################

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

#########################

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

#########################





