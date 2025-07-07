from main import *
# events = load_events("recordings/recording_2025-03-26_13-41-19.raw")
# freq = 5.5  # or use: freq = estimate_frequency(events)
# cycles = extract_full_cycles(events, freq)
# T = int(1e6 / freq)
# mega_wave = fold_events(cycles, T)
# plot_mega_wave(mega_wave, T)


# All that gets replaced by using the analyse function that returns mega_wave,T,freq.
mega_wave, T, freq = analyse("recordings/recording_2025-03-26_13-41-19.raw", freq = 5.5)
