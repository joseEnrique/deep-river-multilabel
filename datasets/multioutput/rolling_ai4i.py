"""
RollingAi4i Dataset - Generates rolling window sequences from Ai4i dataset.

This dataset wrapper takes the Ai4i dataset and yields sequences (DataFrames)
instead of individual instances (dicts). Compatible with DirectMultiLabelForecaster.
"""

import pandas as pd
from .ai4i import Ai4i


class RollingAi4i:
    """
    Rolling window wrapper for Ai4i dataset.

    Yields OVERLAPPING sliding windows of past_size rows as DataFrames.
    Each window slides by 1 instance, creating temporal sequences for LSTM.

    Parameters
    ----------
    past_size : int
        Number of instances to include in each window (sequence length).
    n_instances : int, optional
        Maximum number of WINDOWS to yield. If None, yields all data.
    include_targets : bool, optional
        Whether to include target columns in the DataFrame. Default is False.
        WARNING: Setting this to True causes data leakage!

    Yields
    ------
    tuple
        (x_df, y) where:
        - x_df: pd.DataFrame with past_size rows (features only, NO targets)
        - y: dict with the NEXT instance's target labels (to predict)

    Example
    -------
    >>> from datasets.multioutput import RollingAi4i
    >>> stream = RollingAi4i(past_size=5, n_instances=100)
    >>> for x_df, y in stream:
    ...     print(f"Window shape: {x_df.shape}")
    ...     print(f"Next target: {y}")
    ...     break
    Window shape: (5, 6)  # 5 timesteps, 6 features
    Next target: {'TWF': False, 'HDF': False, ...}

    Note
    ----
    OVERLAPPING windows (sliding window with step=1):
    - Window 1: instances [1,2,3,4,5] → predict instance 6
    - Window 2: instances [2,3,4,5,6] → predict instance 7
    - Window 3: instances [3,4,5,6,7] → predict instance 8

    This creates proper temporal sequences for LSTM training.
    """

    def __init__(self, past_size=1, n_instances=None, include_targets=False, target_mode='forecasting'):
        self.past_size = past_size
        self.n_instances = n_instances
        self.include_targets = include_targets
        self.target_mode = target_mode  # 'forecasting' or 'classification'
        self.target_names = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']

    def __iter__(self):
        """
        Iterate through the dataset yielding OVERLAPPING sliding windows.

        IMPORTANT: Windows OVERLAP (step=1) to create proper temporal sequences.
        With past_size=5:
        - Window 1: [1,2,3,4,5] → predict 6 (forecasting) OR predict 5 (classification)
        """
        # Initialize the base dataset
        dataset = Ai4i()

        # Buffer to collect window instances (sliding window)
        window_buffer = []

        # Track number of windows yielded
        windows_yielded = 0

        for x, y in dataset:
            # NEVER include targets in features (prevent data leakage)
            instance = x.copy()

            # Add current instance to window buffer
            window_buffer.append(instance)

            # Determine buffer size threshold based on mode
            # Classification: need 'past_size' instances (including current) -> predict current
            # Forecasting: need 'past_size' past instances + 1 current instance (target) -> predict current
            threshold = self.past_size if self.target_mode == 'classification' else self.past_size + 1

            # Once we have enough instances, start yielding windows
            if len(window_buffer) == threshold:
                
                if self.target_mode == 'forecasting':
                    # Forecasting: predict NEXT instance using PAST instances
                    # Window: [x_{t-N}, ..., x_{t-1}] -> Target: y_t
                    window = window_buffer[:-1]
                else:
                    # Classification: predict CURRENT instance using CURRENT + PAST instances
                    # Window: [x_{t-N+1}, ..., x_t] -> Target: y_t
                    window = window_buffer[:]
                
                # Convert window to DataFrame
                x_df = pd.DataFrame(window)

                # Target is the NEXT instance (last one in buffer)
                # This is what we want to predict (y_t)
                next_y = {k: v for k, v in y.items() if k in self.target_names}

                # Yield window and next target
                yield x_df, next_y

                windows_yielded += 1

                # Slide window by removing first instance (OVERLAPPING)
                window_buffer.pop(0)

                # Check if we've reached the window limit
                if self.n_instances is not None and windows_yielded >= self.n_instances:
                    break
