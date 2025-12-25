# xdp_knapsack (Python)

Python 3.10+ bindings for the XDP knapsack solvers implemented in Rust.

## Build and install (local)

```bash
cd python
python -m pip install maturin
maturin build --release --out dist
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

## API

- `Item(id: int, profit: float, weight: float)`
- Items can also be tuples or lists `(id, profit, weight)` or `(profit, weight)`, or dicts
  `{\"id\": int, \"profit\": float, \"weight\": float}` (id is optional in dicts/2-tuples).
- `KnapsackResult` fields: `max_profit`, `algorithm_name`, `duration_micros`,
  `selected_items`, `pmax`, `error_bound`
- Functions: `solve_xdp_optimized`, `solve_xdp_optimized_with_selection`,
  `approximate_knapsack`, `modified_greedy_pmax`
