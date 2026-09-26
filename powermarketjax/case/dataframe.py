# Vendored from PowerZooJax. Do not edit to track upstream;
# this file is now this repository's code.
# Source        : powerzoojax/case/dataframe.py
# Upstream commit: a7641de (2026-05-07)
# Copied on     : 2026-08-05
# Verbatim copy; only the package name changed (powerzoojax -> powermarketjax,
# in both import statements and path strings).
# Upstream licence: MIT, Copyright (c) 2026 PowerZooJax Contributors.
"""
DataFrame: Extended pandas DataFrame for power system case data

Migrated from PowerZoo's CaseBase.py for use in PowerZooJAX raw cases.
This is a lightweight wrapper around pandas DataFrame with custom display.
"""

from typing import List, Union
import pandas as pd
import numpy as np


class DataFrame(pd.DataFrame):
    """Extended DataFrame for power market data display.
    
    Inherits from pandas.DataFrame with label feature and custom display format.
    
    Example:
        >>> df = DataFrame(
        ...     ['id', 'x', 'y'],
        ...     [[1.0, 0, 0], [2.0, 2, 0]]
        ... )
        >>> df.set_label('Nodes')
        >>> print(df)
    """

    def __init__(self, columns: List[str], data: Union[List, np.ndarray]):
        """Initialize DataFrame.
        
        Args:
            columns: Column names
            data: Data content (2D array or list of lists)
        """
        super(DataFrame, self).__init__(data=data, columns=columns)
        self.loc[:, '#id'] = self.index
        self.index = self['id']
        self.label = ''

    def __repr__(self) -> str:
        """Custom string representation."""
        return f'{"=" * 20} {self.label} {"=" * 20}\n{self.to_string(index=False)}'

    def __str__(self) -> str:
        """String representation."""
        return self.__repr__()

    def set_label(self, label: str) -> 'DataFrame':
        """Set label for display.
        
        Args:
            label: Label name
            
        Returns:
            self to support method chaining
        """
        self.label = label
        return self
