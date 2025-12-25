# xdp_knapsack (Python)

Python 3.10+ bindings for the XDP knapsack solvers implemented in Rust.

## Requirements

- Python 3.10+
- Rust toolchain (stable)
- `maturin>=1.5`

## Install (local)

From the repo root:

```bash
cd python
python -m pip install -U maturin
maturin develop --release
```

To build a wheel instead:

```bash
cd python
maturin build --release --out dist
python -m pip install dist/xdp_knapsack-*.whl
```

## Usage

```python
import xdp_knapsack as xdp

items = [
    xdp.Item(0, 10.0, 5.0),
    (1, 7.0, 3.0),
    {"profit": 8.0, "weight": 4.0},
]

capacity = 8.0
result = xdp.solve_xdp_optimized_with_selection(items, capacity)
print(result.max_profit, result.selected_items)

approx = xdp.approximate_knapsack(items, capacity)
print(approx.max_profit, approx.error_bound)
```

## Inputs

- `Item(id: int, profit: float, weight: float)`
- Tuples/lists: `(profit, weight)` or `(id, profit, weight)`
- Dicts: `{"profit": float, "weight": float}` with optional `id`
- If `id` is omitted, the input index is used (0-based)

## Results

`KnapsackResult` exposes:

- `max_profit`: best profit found
- `selected_items`: list of selected item IDs (empty if selection is not tracked)
- `algorithm_name`: algorithm label for the result
- `duration_micros`: elapsed time in microseconds
- `pmax`: optional fractional upper bound used for error reporting
- `error_bound`: optional relative error bound in `[0, 1]`

## Functions

- `solve_xdp_optimized`: optimized XDP solver (profit only)
- `solve_xdp_optimized_with_selection`: optimized XDP solver with backtracking
- `approximate_knapsack`: XDP approximation with error bound
- `modified_greedy_pmax`: greedy heuristic with fractional upper bound
