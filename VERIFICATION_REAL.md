# Verification record

Code delivered on 2026-09-09.

- Original simulator: 8 tests passed.
- Real-data pipeline: 5 tests passed; 1 optional Parquet test skipped because pyarrow is unavailable in the execution environment.
- Tested cleaning, chronological split rejection, independence of historical travel tables from test trip durations, optimized solver agreement, and a fixture-based path through parameter selection, simulation, and LaTeX reporting.
- Synthetic fixtures are identified and rejected by production data loading/reporting; test export is explicitly labeled.
- Configuration and command-line entry point checked.
- Available test interpreter: Python in the user's Anaconda installation; pandas 2.3.3, NumPy 1.24.0, SciPy 1.15.3.
- Dependency installation was attempted in an isolated temporary environment but package-server DNS/network access was unavailable. The temporary environment was removed; existing environments were not changed.
- The actual TLC monthly Parquet was NOT downloaded or run here. Official source URLs and field definitions were checked on the NYC TLC website. Local installation of requirements-real.txt and execution of download/prepare/smoke are still needed.
- No actual research results, selected weights, or effectiveness claims are supplied with this delivery.
