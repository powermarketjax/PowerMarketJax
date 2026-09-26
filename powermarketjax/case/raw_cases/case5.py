# Vendored from PowerZooJax. Do not edit to track upstream;
# this file is now this repository's code.
# Source        : powerzoojax/case/raw_cases/case5.py
# Upstream commit: a7641de (2026-05-07)
# Copied on     : 2026-08-05
# Verbatim copy; only the package name changed (powerzoojax -> powermarketjax,
# in both import statements and path strings).
# Upstream licence: MIT, Copyright (c) 2026 PowerZooJax Contributors.
"""
Case5: 5-Bus Test System (Original OOP Style)

Migrated from PowerZoo with minimal changes:
- Removed ClearCase inheritance
- Uses local DataFrame import
- Self-contained definition
"""

from powermarketjax.case.dataframe import DataFrame


class Case5:
    """5-Bus transmission test case.
    
    Network topology:
        1 ------- 2
        | \\       |
        |  \\      |
        |   \\     |
        5- - 4 -- 3
    
    Generators at buses: 1, 1, 3, 4, 5
    Loads at buses: 1, 2, 3, 4, 5
    """
    
    def __init__(self):
        """Initialize Case5 with nodes, units, lines, and loads."""
        
        self.nodes = DataFrame(
            ['id', 'x', 'y'],
            [[1.0, 0, 0],
             [2.0, 2, 0],
             [3.0, 4, 2],
             [4.0, 2, 2],
             [5.0, 0, 2]]
        )

        self.units = DataFrame(
            ['id', 'bus_id', 'mc_a', 'mc_b', 'mc_c', 'p_max', 'p_min'],
            [[1.0, 1.0, 0.0, 0.0, 14.0, 40.0, 5.0],
             [2.0, 1.0, 0.0, 0.0, 15.0, 170.0, 10.0],
             [3.0, 3.0, 0.0, 0.0, 30.0, 520.0, 20.0],
             [4.0, 4.0, 0.0, 0.0, 40.0, 200.0, 10.0],
             [5.0, 5.0, 0.0, 0.0, 10.0, 600.0, 20.0]]
        )

        self.lines = DataFrame(
            ['id', 'from', 'to', 'x', 'floor', 'cap'],
            [[1.0, 1.0, 2.0, 0.0281, -400.0, 400.0],
             [2.0, 1.0, 4.0, 0.0304, 0.0, 0.0],
             [3.0, 1.0, 5.0, 0.0064, 0.0, 0.0],
             [4.0, 2.0, 3.0, 0.0108, 0.0, 0.0],
             [5.0, 3.0, 4.0, 0.0297, 0.0, 0.0],
             [6.0, 4.0, 5.0, 0.0297, -240.0, 240.0]]
        )

        self.loads = DataFrame(
            ['id', 'bus_id', 'mc_a', 'mc_b', 'mc_c', 'd_max', 'd_min'],
            [[1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             [2.0, 2.0, 0.0, 0.0, 0.0, 500.0, 300.0],
             [3.0, 3.0, 0.0, 0.0, 0.0, 600.0, 300.0],
             [4.0, 4.0, 0.0, 0.0, 0.0, 400.0, 400.0],
             [5.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
        )
        
        # Fix zero limits (convert to large values)
        self.lines.loc[self.lines['cap'] == 0, 'cap'] = 1000000
        self.lines.loc[self.lines['floor'] == 0, 'floor'] = -1000000
        
        # Set name
        self.name = 'Case5'


if __name__ == '__main__':
    case = Case5()
    print(f"Case: {case.name}")
    print(f"Nodes: {len(case.nodes)}")
    print(f"Units: {len(case.units)}")
    print(f"Lines: {len(case.lines)}")
    print(f"Loads: {len(case.loads)}")
    print("\nNodes:")
    print(case.nodes)
    print("\nUnits:")
    print(case.units)
