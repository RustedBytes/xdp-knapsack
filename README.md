# xdp_knapsack

Experimental 0/1 knapsack implementations in Rust, focused on an optimized XDP
approximation plus supporting heuristics.

## Features
- Optimized XDP solver with optional item selection backtracking.
- XDP approximation with error bound based on a greedy fractional upper bound.
- Modified greedy heuristic that estimates Pmax.

## Crate Usage
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

## Build
```bash
cargo build --release
```
