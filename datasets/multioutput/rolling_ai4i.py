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
    
    Yields sequences of past_size rows as DataFrames, suitable for
    DirectMultiLabelForecaster which expects DataFrame inputs.
    
    Parameters
    ----------
    past_size : int
        Number of past instances to include in each sequence window.
    n_instances : int, optional
        Maximum number of instances to yield. If None, yields all instances.
    include_targets : bool, optional
        Whether to include target columns in the DataFrame. Default is False.
        
    Yields
    ------
    tuple
        (x_df, y) where:
        - x_df: pd.DataFrame with past_size rows (features + optional targets)
        - y: dict with current target labels
    
    Example
    -------
    >>> from datasets.multioutput import RollingAi4i
    >>> stream = RollingAi4i(past_size=5, n_instances=100, include_targets=True)
    >>> for x_df, y in stream:
    ...     print(f"Sequence shape: {x_df.shape}")
    ...     print(f"Targets: {y}")
    ...     break
    Sequence shape: (5, 11)  # 5 timesteps, 11 features
    Targets: {'TWF': False, 'HDF': False, ...}
    """
    
    def __init__(self, past_size=1, n_instances=None, include_targets=False):
        self.past_size = past_size
        self.n_instances = n_instances
        self.include_targets = include_targets
        self.target_names = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
        
    def __iter__(self):
        """
        Iterate through the dataset yielding rolling window sequences.
        """
        # Initialize the base dataset
        dataset = Ai4i()
        
        # Buffer to store past instances
        buffer = []
        
        # Track number of instances yielded
        count = 0
        
        for x, y in dataset:
            # Add current instance to buffer
            # Combine features and optionally targets
            if self.include_targets:
                instance = {**x, **y}
            else:
                instance = x
            
            buffer.append(instance)
            
            # Keep buffer at most past_size
            if len(buffer) > self.past_size:
                buffer.pop(0)
            
            # Only yield when buffer is full
            if len(buffer) == self.past_size:
                # Convert buffer to DataFrame
                x_df = pd.DataFrame(buffer)
                
                # Yield sequence and current target
                yield x_df, y
                
                count += 1
                
                # Check if we've reached the limit
                if self.n_instances is not None and count >= self.n_instances:
                    break
