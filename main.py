from faery.events_input import events_stream_from_file
import numpy as np
import matplotlib.pyplot as plt

#Remember, all files must be saved in the recordings directory that is at the same level as other codes and venv and external libraries.
# Always pip install faery to use this code.

def load_events(path):
    """
    Load event stream from a .raw file.
    """
    decoder = events_stream_from_file(path)
    return decoder.to_array()


def estimate_frequency(events, max_time=None):
    """
    Estimate dominant event frequency using ON/OFF transitions.
    """
    t = events["t"]
    p = events["on"]
    if max_time:
        t, p = t[t < max_time], p[t < max_time]

    changes = np.flatnonzero(np.diff(p.astype(int)))
    intervals = np.diff(t[changes])
    median_period = np.median(intervals)
    return 1e6 / median_period if median_period > 0 else 1.0


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

def analyse(path, freq=None):
    """
    Load events, extract full cycles, and fold into a single mega-wave.

    Parameters:
        path (str): Path to the raw event file.
        freq (float, optional): Frequency in Hz. If None, it will be estimated.

    Returns:
        tuple: (mega_wave_dict, T, freq)
            - mega_wave_dict: dict with folded 't', 'x', 'y', 'on' arrays
            - T: period in microseconds
            - freq: frequency used in analysis
    """
    events = load_events(path)

    if freq is None:
        freq = estimate_frequency(events)

    T = int(1e6 / freq)
    cycles = extract_full_cycles(events, freq)
    mega_wave = fold_events(cycles, T)

    return mega_wave, T, freq
