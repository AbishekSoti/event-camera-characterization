from faery.events_input import events_stream_from_file
import numpy as np
import matplotlib.pyplot as plt


# Remember, all files must be saved in the recordings directory that is at the same level as other codes and venv and external libraries.
# Always pip install faery to use this code.

def load_events(path):
    """
    Load event stream from a .raw file.
    Then convert it to an array using Faery's under-the-hood decoder object.
    This stores values in decoder in the format of t,x,y,on which can be
    referenced via decoder["t"], decoder["on"], and so on.
    """
    decoder = events_stream_from_file(path)
    return decoder.to_array()


def find_clean_start(events, T, verbose=False):
    """
    Find stable 3x ON or OFF sequence within half a period to set t0.
    """
    t, p = events["t"], events["on"]
    for i in range(2, len(p)):
        if p[i] == p[i - 1] == p[i - 2] and (t[i] - t[i - 2]) < (T // 2):
            t0 = t[i]
            if verbose:
                print(f"[DEBUG] Found clean start at t = {t0 / 1e6:.6f} s, polarity = {p[i]}")
            return t0
    if verbose:
        print(f"[DEBUG] No clean start found, falling back to t[0] = {t[0] / 1e6:.6f} s")
    return t[0]


def extract_full_cycles(events, T, verbose=False):
    """
    Drop partial cycles and extract full periods using a stable starting point.

    Parameters:
        events: np.ndarray of events
        T: Period in microseconds
        verbose: Print debug information

    Returns:
        List of full cycles (each a structured array of events)
    """
    t = events["t"]
    t0 = find_clean_start(events, T, verbose=verbose)
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

    Parameters:
        cycles: List of full cycles
        T: Period in microseconds

    Returns:
        Dict containing concatenated folded arrays ('t', 'x', 'y', 'on')
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

    if verbose:
        print(f"[DEBUG] Recording starts at {events['t'][0] / 1e6:.6f} s, ends at {events['t'][-1] / 1e6:.6f} s")

    T = int(1e6 / freq)

    if verbose:
        print(f"[INFO] Using frequency: {freq:.2f} Hz (Period T = {T} µs)")
        print("[INFO] Extracting full cycles...")

    cycles = extract_full_cycles(events, T, verbose=verbose)

    if verbose:
        print(f"[INFO] Extracted {len(cycles)} cycles")
        print("[INFO] Folding events...")

    mega_wave = fold_events(cycles, T)

    if verbose:
        print("[INFO] Done.")

    return mega_wave, T, freq
