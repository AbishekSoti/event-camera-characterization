from faery.events_input import events_stream_from_file
import numpy as np
import matplotlib.pyplot as plt

#Remember, all files must be saved in the recordings directory that is at the same level as other codes and venv and external libraries.
# Always pip install faery to use this code.

def load_events(path):
    """
    Load event stream from a .raw file.
    Then convert it to an array using Faery's under the hood decoder object.
    This stores values in decoder in the format of t,x,y,on which can be
    referenced via decoder["t"],decoder["on"] and such.
    """
    decoder = events_stream_from_file(path)
    return decoder.to_array()

#TODO: Implement a frequency estimator in cause frequency is not known.

# def estimate_frequency_fft(events, bin_size_us=1000, max_time_us=None, plot=False):
#     t = events["t"]
#
#     if max_time_us:
#         t = t[t < max_time_us]
#
#     if len(t) < 2:
#         print("Warning: Not enough events for frequency estimation.")
#         return 1.0
#
#     duration_us = t[-1] - t[0]
#     num_bins = max(1, int(duration_us // bin_size_us))  # avoid zero bins
#
#     hist, _ = np.histogram(t, bins=num_bins)
#     rate = hist.astype(float)
#
#     rate -= np.mean(rate)
#
#     fft_vals = np.fft.fft(rate)
#     freqs = np.fft.fftfreq(len(rate), d=bin_size_us * 1e-6)
#
#     pos_freqs = freqs[freqs > 0]
#     magnitude = np.abs(fft_vals[freqs > 0])
#
#     if len(magnitude) == 0:
#         print("Warning: FFT returned no valid frequency components.")
#         return 1.0
#
#     dominant_idx = np.argmax(magnitude)
#     return pos_freqs[dominant_idx]
#

def find_clean_start(events, T):
    """
    Find stable 3x ON or OFF sequence within half a period to set t0.
    """
    t, p = events["t"], events["on"]
    for i in range(2, len(p)):
        if p[i] == p[i-1] == p[i-2] and (t[i] - t[i-2]) < (T // 2):
            return t[i]
    return t[0]  # fallback


def extract_full_cycles(events, freq):
    """
    Drop partial cycles and extract full periods using a stable starting point.
    """
    T = int(1e6 / freq)  # µs
    t = events["t"]
    t0 = find_clean_start(events, T)
    t_end = t[-1]
    cycles = []

    while t0 + T <= t_end:
        mask = (t >= t0) & (t < t0 + T)
        cycles.append(events[mask])
        t0 += T

    return cycles


def fold_events(cycles, T):
    """
    Fold events from multiple full cycles into a single reference period.
    """
    mega_t, mega_x, mega_y, mega_on = [], [], [], []

    for cycle in cycles:
        folded_t = (cycle["t"] - cycle["t"][0]) % T
        mega_t.append(folded_t)
        mega_x.append(cycle["x"])
        mega_y.append(cycle["y"])
        mega_on.append(cycle["on"])

    return {
        "t": np.concatenate(mega_t),
        "x": np.concatenate(mega_x),
        "y": np.concatenate(mega_y),
        "on": np.concatenate(mega_on)
    }

# Ignore plots for now.
def plot_mega_wave(mega_wave, T, bins=200):
    """
    Plot histogram of ON and OFF events over one folded cycle.
    """
    t = mega_wave["t"]
    on = mega_wave["on"]

    plt.figure(figsize=(10, 4))
    plt.hist(t[on], bins=bins, range=(0, T), alpha=0.6, label="ON events")
    plt.hist(t[~on], bins=bins, range=(0, T), alpha=0.6, label="OFF events")
    plt.title("Phase-Aligned Mega Wave")
    plt.xlabel("Time within Cycle (µs)")
    plt.ylabel("Event Count")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def analyse(path, freq, verbose=True):
    """
    Load events, extract full cycles, and fold into a single mega-wave.

    Parameters:
        path (str): Path to the raw event file.
        freq (float): Frequency in Hz (required).
        verbose (bool): Whether to print diagnostic information.

    Returns:
        tuple: (mega_wave_dict, T, freq)
            - mega_wave_dict: dict with folded 't', 'x', 'y', 'on' arrays
            - T: period in microseconds
            - freq: frequency used in analysis
    """
    if freq is None:
        raise ValueError("Frequency must be provided.")

    if verbose:
        print("[INFO] Loading events...")

    events = load_events(path)

    T = int(1e6 / freq)

    if verbose:
        print(f"[INFO] Using frequency: {freq:.2f} Hz (Period T = {T} µs)")
        print("[INFO] Extracting full cycles...")

    cycles = extract_full_cycles(events, freq)

    if verbose:
        print(f"[INFO] Extracted {len(cycles)} cycles")
        print("[INFO] Folding events...")

    mega_wave = fold_events(cycles, T)

    if verbose:
        print("[INFO] Done.")

    return mega_wave, T, freq
