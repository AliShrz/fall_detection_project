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

    print(f"\nDone ✅")
    print(f"Total files : {counter}")
    print(f"ADL         : {adl_count}")
    print(f"Fall        : {fall_count}")

    return all_data, all_labels, activity_codes, file_names

########################

