use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyAnyMethods;
use pyo3::types::{PyDict, PyDictMethods, PySequence, PySequenceMethods};

use ::xdp_knapsack as core;

#[pyclass(name = "Item")]
#[derive(Clone)]
struct PyItem {
    #[pyo3(get, set)]
    id: usize,
    #[pyo3(get, set)]
    profit: f64,
    #[pyo3(get, set)]
    weight: f64,
}

#[pymethods]
impl PyItem {
    #[new]
    fn new(id: usize, profit: f64, weight: f64) -> Self {
        Self { id, profit, weight }
    }
}

impl From<PyItem> for core::Item {
    fn from(item: PyItem) -> Self {
        Self {
            id: item.id,
            profit: item.profit,
            weight: item.weight,
        }
    }
}

#[derive(Clone)]
struct PyItemInput {
    id: Option<usize>,
    profit: f64,
    weight: f64,
}

impl<'py> FromPyObject<'py> for PyItemInput {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        if let Ok(item) = ob.extract::<PyRef<PyItem>>() {
            return Ok(Self {
                id: Some(item.id),
                profit: item.profit,
                weight: item.weight,
            });
        }

        if let Ok(dict) = ob.downcast::<PyDict>() {
            let profit = dict
                .get_item("profit")?
                .ok_or_else(|| PyValueError::new_err("Item dict missing 'profit'"))?
                .extract::<f64>()?;
            let weight = dict
                .get_item("weight")?
                .ok_or_else(|| PyValueError::new_err("Item dict missing 'weight'"))?
                .extract::<f64>()?;
            let id = match dict.get_item("id")? {
                Some(value) => Some(value.extract::<usize>()?),
                None => None,
            };
            return Ok(Self { id, profit, weight });
        }

        if let Ok(seq) = ob.downcast::<PySequence>() {
            let len = seq.len()?;
            if len == 2 {
                let profit = seq.get_item(0)?.extract::<f64>()?;
                let weight = seq.get_item(1)?.extract::<f64>()?;
                return Ok(Self {
                    id: None,
                    profit,
                    weight,
                });
            }
            if len == 3 {
                let id = seq.get_item(0)?.extract::<usize>()?;
                let profit = seq.get_item(1)?.extract::<f64>()?;
                let weight = seq.get_item(2)?.extract::<f64>()?;
                return Ok(Self {
                    id: Some(id),
                    profit,
                    weight,
                });
            }
            return Err(PyValueError::new_err(
                "Item sequence must have length 2 (profit, weight) or 3 (id, profit, weight)",
            ));
        }

        Err(PyValueError::new_err(
            "Item must be Item, dict, or sequence",
        ))
    }
}

#[pyclass(name = "KnapsackResult")]
struct PyKnapsackResult {
    #[pyo3(get)]
    max_profit: f64,
    #[pyo3(get)]
    algorithm_name: String,
    #[pyo3(get)]
    duration_micros: u128,
    #[pyo3(get)]
    selected_items: Vec<usize>,
    #[pyo3(get)]
    pmax: Option<f64>,
    #[pyo3(get)]
    error_bound: Option<f64>,
}

impl From<core::KnapsackResult> for PyKnapsackResult {
    fn from(result: core::KnapsackResult) -> Self {
        Self {
            max_profit: result.max_profit,
            algorithm_name: result.algorithm_name,
            duration_micros: result.duration_micros,
            selected_items: result.selected_items,
            pmax: result.pmax,
            error_bound: result.error_bound,
        }
    }
}

fn to_core_items(items: Vec<PyItemInput>) -> Vec<core::Item> {
    items
        .into_iter()
        .enumerate()
        .map(|(idx, item)| core::Item {
            id: item.id.unwrap_or(idx),
            profit: item.profit,
            weight: item.weight,
        })
        .collect()
}

#[pyfunction]
fn solve_xdp_optimized(items: Vec<PyItemInput>, capacity: f64) -> PyResult<PyKnapsackResult> {
    let result = core::solve_xdp_optimized(to_core_items(items), capacity);
    Ok(result.into())
}

#[pyfunction]
fn solve_xdp_optimized_with_selection(
    items: Vec<PyItemInput>,
    capacity: f64,
) -> PyResult<PyKnapsackResult> {
    let result = core::solve_xdp_optimized_with_selection(to_core_items(items), capacity);
    Ok(result.into())
}

#[pyfunction]
fn approximate_knapsack(items: Vec<PyItemInput>, capacity: f64) -> PyResult<PyKnapsackResult> {
    match core::approximate_knapsack(to_core_items(items), capacity) {
        Ok(result) => Ok(result.into()),
        Err(message) => Err(PyValueError::new_err(message)),
    }
}

#[pyfunction]
fn modified_greedy_pmax(items: Vec<PyItemInput>, capacity: f64) -> PyResult<PyKnapsackResult> {
    let result = core::modified_greedy_pmax(to_core_items(items), capacity);
    Ok(result.into())
}

#[pymodule]
fn xdp_knapsack(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<PyItem>()?;
    m.add_class::<PyKnapsackResult>()?;
    m.add_function(wrap_pyfunction!(solve_xdp_optimized, m)?)?;
    m.add_function(wrap_pyfunction!(solve_xdp_optimized_with_selection, m)?)?;
    m.add_function(wrap_pyfunction!(approximate_knapsack, m)?)?;
    m.add_function(wrap_pyfunction!(modified_greedy_pmax, m)?)?;
    Ok(())
}
