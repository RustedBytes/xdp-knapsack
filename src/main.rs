use xdp_knapsack::{
    approximate_knapsack, modified_greedy_pmax, solve_xdp_optimized_with_selection, Item,
};

fn main() {
    // Generate large dataset to demonstrate sorting speedup
    let n = 1_000_000;
    let capacity = 500_000.0;

    println!("Generating {} items...", n);
    let mut items = Vec::with_capacity(n);
    for i in 0..n {
        items.push(Item {
            id: i,
            profit: (i as f64 % 100.0) + 1.0,
            weight: (i as f64 % 50.0) + 1.0,
        });
    }

    /*
    let n = 5;
    let capacity = 100.0;

    println!("Generating {} items...", n);
    let mut items = Vec::with_capacity(n);

    items.push(Item {
        id: 0,
        profit: 2.0,
        weight: 1.0,
    });
    items.push(Item {
        id: 1,
        profit: 100.0,
        weight: 100.0,
    });
    items.push(Item {
        id: 2,
        profit: 50.0,
        weight: 30.0,
    });
    items.push(Item {
        id: 3,
        profit: 60.0,
        weight: 40.0,
    });
    items.push(Item {
        id: 4,
        profit: 70.0,
        weight: 50.0,
    });
    */

    println!("-----------------------------------");

    println!("Starting XDP Optimized...");
    let result = solve_xdp_optimized_with_selection(items.clone(), capacity);

    println!("Max Profit: {:.2}", result.max_profit);
    println!("Time: {} µs", result.duration_micros);
    if let Some(pmax) = result.pmax {
        println!("Pmax (Greedy): {:.2}", pmax);
    }
    if let Some(error_bound) = result.error_bound {
        println!("Error Bound: {:.6}", error_bound);
    }
    println!("Selected Items: {:?}", result.selected_items.len());

    println!("-----------------------------------");

    println!("Starting Approximate Knapsack...");
    match approximate_knapsack(items.clone(), capacity) {
        Ok(result) => {
            println!("Max Profit: {:.2}", result.max_profit);
            println!("Time: {} µs", result.duration_micros);
            if let Some(pmax) = result.pmax {
                println!("Pmax (Greedy): {:.2}", pmax);
            }
            if let Some(error_bound) = result.error_bound {
                println!("Error Bound: {:.6}", error_bound);
            }
            println!("Selected Items: {:?}", result.selected_items.len());
        }
        Err(e) => eprintln!("Error: {}", e),
    }

    println!("-----------------------------------");

    println!("Start modified_greedy_pmax...");
    let result = modified_greedy_pmax(items.clone(), capacity);
    println!("Max Profit: {:.2}", result.max_profit);
    println!("Time: {} µs", result.duration_micros);
    println!("Selected Items: {:?}", result.selected_items.len());
}
