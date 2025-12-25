# xdp_knapsack

Experimental 0/1 knapsack implementations in Rust, focused on optimized XDP
approximations plus supporting heuristics.

Paper: An O(n log n) approximate knapsack algorithm (Nick Dawes).

## Features
- Optimized XDP solver with optional item selection backtracking and a parallel best-bin scan.
- XDP approximation with an error bound derived from a greedy fractional upper bound (Pmax).
- Modified greedy heuristic for fast baselines and Pmax estimation.
- Designed for large instances with float weights and profits.

## Algorithms
- `solve_xdp_optimized`: optimized XDP solver that returns profit only.
- `solve_xdp_optimized_with_selection`: optimized XDP with backtracking; uses O(n * T) memory.
  Selection is auto-disabled if bins exceed 65535 or the backtracking table would overflow.
- `approximate_knapsack`: XDP approximation with error bound reporting.
- `modified_greedy_pmax`: greedy heuristic with fractional upper bound.

## Data Model
`Item` is a simple tuple of identifiers and floating point values:

```
Item { id: usize, profit: f64, weight: f64 }
```

`KnapsackResult` exposes:
- `max_profit`: best profit found
- `selected_items`: list of selected item IDs (empty if selection tracking is disabled)
- `algorithm_name`: algorithm label
- `duration_micros`: elapsed time in microseconds
- `pmax`: optional fractional upper bound used for error reporting
- `error_bound`: optional relative error bound in `[0, 1]`

## Crate Usage
If you are using this repository as a path dependency:

```toml
xdp_knapsack = { path = "." }
```

```rust
use xdp_knapsack::{approximate_knapsack, solve_xdp_optimized_with_selection, Item};

let items = vec![
    Item { id: 0, profit: 10.0, weight: 5.0 },
    Item { id: 1, profit: 7.0, weight: 3.0 },
    Item { id: 2, profit: 8.0, weight: 4.0 },
];

let capacity = 8.0;
let result = solve_xdp_optimized_with_selection(items.clone(), capacity);
println!("profit={:.2} items={:?}", result.max_profit, result.selected_items);

let approx = approximate_knapsack(items, capacity).expect("approx failed");
println!("profit={:.2} err={:?}", approx.max_profit, approx.error_bound);
```

## Demo Binary
The demo in `src/main.rs` generates a synthetic dataset and runs each solver.

```bash
cargo run --release
```

## Python Bindings
Python 3.10+ bindings live in `python/README.md`.

## Build
```bash
cargo build --release
```

## License

MIT License. See `LICENSE` file for details.
