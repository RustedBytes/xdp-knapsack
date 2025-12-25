use rayon::prelude::*;
use std::cmp::Ordering;
use std::time::Instant;

const PAR_BEST_SCAN_MIN_BINS: usize = 4096;

// --- Data Structures ---

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Item {
    pub id: usize,
    pub profit: f64,
    pub weight: f64,
}

#[derive(Debug)]
pub struct KnapsackResult {
    pub max_profit: f64,
    pub algorithm_name: String,
    pub duration_micros: u128,
    pub selected_items: Vec<usize>,
    pub pmax: Option<f64>,
    pub error_bound: Option<f64>,
}

// --- Optimized XDP Solver ---

pub fn solve_xdp_optimized(items: Vec<Item>, capacity: f64) -> KnapsackResult {
    solve_xdp_optimized_internal(items, capacity, false)
}

/// Optimized XDP solver that also tracks selected items.
/// Note: tracking adds O(n * T) memory for backtracking.
pub fn solve_xdp_optimized_with_selection(items: Vec<Item>, capacity: f64) -> KnapsackResult {
    solve_xdp_optimized_internal(items, capacity, true)
}

fn solve_xdp_optimized_internal(
    items: Vec<Item>,
    capacity: f64,
    track_selection: bool,
) -> KnapsackResult {
    let start = Instant::now();
    let n = items.len();

    if n == 0 || capacity <= 0.0 {
        return KnapsackResult {
            max_profit: 0.0,
            algorithm_name: "XDP Optimized".to_string(),
            duration_micros: start.elapsed().as_micros(),
            selected_items: Vec::new(),
            pmax: Some(0.0),
            error_bound: Some(0.0),
        };
    }

    let greedy = modified_greedy_pmax(items.clone(), capacity);
    let pmax = greedy.pmax.unwrap_or(greedy.max_profit);

    // Heuristic parameter T ~ 12 * ln(n)
    let g = 12.0;
    let t_val = (n as f64).ln() * g;
    let num_bins = t_val.ceil().max(1.0) as usize;
    let bins = num_bins + 1;

    // --- Optimization: Double Buffering ---
    // Instead of one array iterated backwards, we use two arrays.
    // This improves CPU cache prefetching (forward iteration) and
    // eliminates Read-After-Write hazards.

    // 'curr' represents state after item i-1
    // 'next' represents state after item i
    let mut curr_p = vec![0.0f64; bins];
    let mut curr_w = vec![0.0f64; bins];
    let mut curr_valid = vec![0u8; bins];
    curr_valid[0] = 1;

    let mut next_p = curr_p.clone();
    let mut next_w = curr_w.clone();
    let mut next_valid = curr_valid.clone();

    // Track active bins to avoid iterating 0..num_bins every time.
    // Initially, only bin 0 is active.
    let mut max_active_bin = 0;

    let mut track_selection = track_selection;
    let mut back: Vec<u16> = Vec::new();
    if track_selection {
        let back_len = n.checked_mul(bins);
        if num_bins > u16::MAX as usize || back_len.is_none() {
            track_selection = false;
        } else {
            back = vec![u16::MAX; back_len.unwrap()];
        }
    }

    // Scaling factor for mapping weight to bins
    let scale_factor = (num_bins as f64) / capacity;

    if track_selection {
        for (i, item) in items.iter().enumerate() {
            let p_i = item.profit;
            let w_i = item.weight;

            let copy_len = max_active_bin + 1;
            next_p[..copy_len].copy_from_slice(&curr_p[..copy_len]);
            next_w[..copy_len].copy_from_slice(&curr_w[..copy_len]);
            next_valid[..copy_len].copy_from_slice(&curr_valid[..copy_len]);

            let mut new_max_active = max_active_bin;

            for j in 0..=max_active_bin {
                if curr_valid[j] == 0 {
                    continue;
                }

                let w_new = curr_w[j] + w_i;
                if w_new > capacity {
                    continue;
                }

                let p_new = curr_p[j] + p_i;
                let mut k = (w_new * scale_factor) as usize;
                if k > num_bins {
                    k = num_bins;
                }

                if next_valid[k] == 0 || p_new > next_p[k] {
                    next_p[k] = p_new;
                    next_w[k] = w_new;
                    next_valid[k] = 1;
                    back[i * bins + k] = j as u16;

                    if k > new_max_active {
                        new_max_active = k;
                    }
                }
            }

            std::mem::swap(&mut curr_p, &mut next_p);
            std::mem::swap(&mut curr_w, &mut next_w);
            std::mem::swap(&mut curr_valid, &mut next_valid);

            max_active_bin = new_max_active;
        }
    } else {
        for item in items.iter() {
            let p_i = item.profit;
            let w_i = item.weight;

            let copy_len = max_active_bin + 1;
            next_p[..copy_len].copy_from_slice(&curr_p[..copy_len]);
            next_w[..copy_len].copy_from_slice(&curr_w[..copy_len]);
            next_valid[..copy_len].copy_from_slice(&curr_valid[..copy_len]);

            let mut new_max_active = max_active_bin;

            for j in 0..=max_active_bin {
                if curr_valid[j] == 0 {
                    continue;
                }

                let w_new = curr_w[j] + w_i;
                if w_new > capacity {
                    continue;
                }

                let p_new = curr_p[j] + p_i;
                let mut k = (w_new * scale_factor) as usize;
                if k > num_bins {
                    k = num_bins;
                }

                if next_valid[k] == 0 || p_new > next_p[k] {
                    next_p[k] = p_new;
                    next_w[k] = w_new;
                    next_valid[k] = 1;

                    if k > new_max_active {
                        new_max_active = k;
                    }
                }
            }

            std::mem::swap(&mut curr_p, &mut next_p);
            std::mem::swap(&mut curr_w, &mut next_w);
            std::mem::swap(&mut curr_valid, &mut next_valid);

            max_active_bin = new_max_active;
        }
    }

    let scan_len = max_active_bin + 1;
    let (best_bin, best_profit) = if scan_len >= PAR_BEST_SCAN_MIN_BINS {
        curr_p[..scan_len]
            .par_iter()
            .enumerate()
            .filter_map(|(i, &p)| {
                if curr_valid[i] != 0 {
                    Some((i, p))
                } else {
                    None
                }
            })
            .reduce_with(|a, b| if a.1 >= b.1 { a } else { b })
            .unwrap_or((0, 0.0))
    } else {
        let mut best_profit = 0.0;
        let mut best_bin = 0;
        for i in 0..scan_len {
            if curr_valid[i] != 0 && curr_p[i] > best_profit {
                best_profit = curr_p[i];
                best_bin = i;
            }
        }
        (best_bin, best_profit)
    };

    let selected_items = if track_selection && best_profit > 0.0 {
        let mut selected = Vec::new();
        let mut k = best_bin;
        for i in (0..n).rev() {
            let prev = back[i * bins + k];
            if prev != u16::MAX {
                selected.push(items[i].id);
                k = prev as usize;
            }
        }
        selected.reverse();
        selected
    } else {
        Vec::new()
    };

    let error_bound = if pmax > 0.0 {
        ((pmax - best_profit) / pmax).max(0.0)
    } else {
        0.0
    };

    let duration = start.elapsed().as_micros();
    KnapsackResult {
        max_profit: best_profit,
        algorithm_name: "XDP Optimized".to_string(),
        duration_micros: duration,
        selected_items,
        pmax: Some(pmax),
        error_bound: Some(error_bound),
    }
}

/// Solves the 0/1 Knapsack problem using the XDP algorithm (Dawes, 2025).
///
/// # Arguments
/// * `items` - A list of items with profit and weight.
/// * `capacity` - The maximum weight capacity of the knapsack.
///
/// # Returns
/// * `Result<KnapsackResult, String>` - The optimal subset approximation.
pub fn approximate_knapsack(items: Vec<Item>, capacity: f64) -> Result<KnapsackResult, String> {
    let start = Instant::now();

    if items.is_empty() || capacity <= 0.0 {
        return Ok(KnapsackResult {
            max_profit: 0.0,
            selected_items: Vec::new(),
            algorithm_name: "XDP Approximation".to_string(),
            duration_micros: start.elapsed().as_micros(),
            pmax: Some(0.0),
            error_bound: Some(0.0),
        });
    }

    let n = items.len();
    // Pmax from greedy plus extensions is used to report the approximation error bound.
    let greedy = modified_greedy_pmax(items.clone(), capacity);
    let pmax = greedy.pmax.unwrap_or(greedy.max_profit);

    // 2. Initialize XDP parameters.
    // T is the number of bins. Paper suggests T = ln(n) * 12.
    // We use `ceil` to ensure at least some bins, and strictly > 0.
    let g = 12.0;
    let t_val = (n as f64).ln() * g;
    let num_bins = t_val.ceil() as usize; // T

    // Ensure at least 1 bin to avoid division by zero logic issues.
    let num_bins = if num_bins < 1 { 1 } else { num_bins };

    // DP State Arrays
    // xp[k]: Max profit in bin k
    let mut xp: Vec<f64> = vec![0.0; num_bins + 1];
    // xw[k]: Weight associated with max profit in bin k
    let mut xw: Vec<f64> = vec![0.0; num_bins + 1];
    // xo[k]: Index of the *last item added* to achieve the state in bin k.
    // None indicates the bin is empty.
    let mut xo: Vec<Option<usize>> = vec![None; num_bins + 1];

    // Initialize Bin 0 (Start state)
    xo[0] = Some(usize::MAX); // Sentinel for "start of subset"

    // Backtracking table: back[i][k] stores the `xo` index of the previous state
    // that led to item `i` being placed in bin `k`.
    // Flattened 2D array: index = i * (num_bins + 1) + k.
    let mut back: Vec<Option<usize>> = vec![None; n * (num_bins + 1)];

    let mut best_profit = 0.0;
    let mut best_bin = 0;

    // 3. Main DP Loop
    // Iterate through items (0 to n-1)
    for (i, item) in items.iter().enumerate() {
        // Iterate bins downwards to avoid using the same item twice for the same step
        for j in (0..=num_bins).rev() {
            // If bin j has a valid subset
            if let Some(prev_item_idx) = xo[j] {
                let proposal_profit = xp[j] + item.profit;
                let proposal_weight = xw[j] + item.weight;

                if proposal_weight <= capacity {
                    // Calculate target bin k
                    // k = floor(weight * T / C)
                    let k = (proposal_weight * (num_bins as f64) / capacity).floor() as usize;

                    // Clamp k to be within bounds (floating point safety)
                    let k = k.min(num_bins);

                    // Update if we found a better profit for this target bin
                    if proposal_profit > xp[k] {
                        xp[k] = proposal_profit;
                        xw[k] = proposal_weight;

                        // Record history: The previous item was `prev_item_idx`
                        // stored in `xo[j]`. We store that in `back` for current item `i`.
                        back[i * (num_bins + 1) + k] = Some(prev_item_idx);

                        // Update current bin's last item to be `i`
                        xo[k] = Some(i);

                        // Track global best
                        if proposal_profit > best_profit {
                            best_profit = proposal_profit;
                            best_bin = k;
                        }
                    }
                }
            }
        }
    }

    // 4. Backtracking to reconstruct solution
    let mut selected_indices = Vec::new();

    // Start from the best state
    if best_profit > 0.0 {
        // Get the last item added to the best bin
        let mut current_item_idx_opt = xo[best_bin];
        let mut current_weight = xw[best_bin];

        // Loop until we hit the sentinel (usize::MAX)
        while let Some(curr_idx) = current_item_idx_opt {
            if curr_idx == usize::MAX {
                break;
            }

            // Add the original item ID to the result
            selected_indices.push(items[curr_idx].id);

            // Prepare for next step back
            let item_weight = items[curr_idx].weight;

            // Re-calculate the bin index `k` for the CURRENT state
            // (needed to look up the `back` array)
            // Note: In the forward pass, we put item `curr_idx` into `current_bin`
            // resulting in `current_weight`.
            let k = (current_weight * (num_bins as f64) / capacity).floor() as usize;
            let k = k.min(num_bins);

            // Look up who pointed to this state
            let prev_idx = back[curr_idx * (num_bins + 1) + k];

            // Update weight to the previous state's weight
            current_weight -= item_weight;

            // Move pointer
            current_item_idx_opt = prev_idx;
        }
    }

    let error_bound = if pmax > 0.0 {
        ((pmax - best_profit) / pmax).max(0.0)
    } else {
        0.0
    };

    Ok(KnapsackResult {
        max_profit: best_profit,
        selected_items: selected_indices,
        algorithm_name: "XDP Approximation".to_string(),
        duration_micros: start.elapsed().as_micros(),
        pmax: Some(pmax),
        error_bound: Some(error_bound),
    })
}

/// Greedy-plus approximation for 0/1 knapsack.
/// Greedily fills by density, then computes the fractional upper bound (Pmax)
/// and the maximum fractional error e.
pub fn modified_greedy_pmax(mut items: Vec<Item>, capacity: f64) -> KnapsackResult {
    let start = Instant::now();

    if items.is_empty() || capacity <= 0.0 {
        return KnapsackResult {
            max_profit: 0.0,
            selected_items: Vec::new(),
            algorithm_name: "Modified Greedy".to_string(),
            duration_micros: start.elapsed().as_micros(),
            pmax: Some(0.0),
            error_bound: Some(0.0),
        };
    }

    // Greedy by profit/weight ratio.
    items.sort_by(|a, b| {
        let eff_a = a.profit / a.weight;
        let eff_b = b.profit / b.weight;
        eff_b.partial_cmp(&eff_a).unwrap_or(Ordering::Equal)
    });

    let mut total_weight = 0.0;
    let mut total_profit = 0.0;
    let mut selected = Vec::new();
    let mut pmax: Option<f64> = None;

    for item in items.iter() {
        if total_weight + item.weight <= capacity {
            total_weight += item.weight;
            total_profit += item.profit;
            selected.push(item.id);
        } else if pmax.is_none() {
            let remaining = capacity - total_weight;
            if remaining > 0.0 && item.weight > 0.0 {
                let fractional_profit = remaining * (item.profit / item.weight);
                pmax = Some(total_profit + fractional_profit);
            } else {
                pmax = Some(total_profit);
            }
        }
    }

    let pmax_value = pmax.unwrap_or(total_profit).max(total_profit);
    let error_bound = if pmax_value > 0.0 {
        ((pmax_value - total_profit) / pmax_value).max(0.0)
    } else {
        0.0
    };

    KnapsackResult {
        max_profit: total_profit,
        selected_items: selected,
        algorithm_name: "Modified Greedy".to_string(),
        duration_micros: start.elapsed().as_micros(),
        pmax: Some(pmax_value),
        error_bound: Some(error_bound),
    }
}
