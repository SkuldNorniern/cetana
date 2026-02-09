//! Execution of the tensor op graph on a backend.
//!
//! Runs a [`TensorGraph`] using a [`Backend`]'s primitives (add, multiply, matmul, etc.),
//! with scheduling driven by the graph: nodes in the same [`TensorGraph::parallel_levels`]
//! wave are independent and can run in parallel. Use [`execute_graph`] for sequential
//! execution or [`execute_graph_parallel`] to run each level's nodes concurrently (CPU).
//! See `scheduling_plan.md` in this directory for the full roadmap (memory reuse, fusion, etc.).

use crate::backend::Backend;
use crate::tensor::{ExecutableGraph, Node, NodeId, Op, TensorRef};
use crate::{MlError, MlResult};
use laminax_types::DTypeId;
use std::sync::Arc;

fn buffer_index(input_count: usize, r: TensorRef) -> usize {
    match r {
        TensorRef::Input(i) => i,
        TensorRef::Node(n) => input_count + n.0,
    }
}

fn shape_num_elements(shape: &[usize]) -> usize {
    shape.iter().product()
}

/// Runs a single node given its input buffers and shapes (in order of `node.inputs`).
/// Used internally for both sequential and parallel level execution.
fn execute_one_node(
    backend: &dyn Backend,
    node: &Node,
    input_buffers: &[Vec<f32>],
    input_shapes: &[Vec<usize>],
) -> MlResult<Vec<f32>> {
    let result: Vec<f32> = match &node.op {
        Op::Add => {
            if node.inputs.len() != 2 {
                return Err(MlError::StringError("Add expects 2 inputs".to_string()));
            }
            backend.add(&input_buffers[0], &input_buffers[1])
        }
        Op::Sub => {
            if node.inputs.len() != 2 {
                return Err(MlError::StringError("Sub expects 2 inputs".to_string()));
            }
            backend.sub(&input_buffers[0], &input_buffers[1])
        }
        Op::Mul => {
            if node.inputs.len() != 2 {
                return Err(MlError::StringError("Mul expects 2 inputs".to_string()));
            }
            backend.multiply(&input_buffers[0], &input_buffers[1])
        }
        Op::Div => {
            if node.inputs.len() != 2 {
                return Err(MlError::StringError("Div expects 2 inputs".to_string()));
            }
            backend.div(&input_buffers[0], &input_buffers[1])
        }
        Op::MatMul => {
            if node.inputs.len() != 2 {
                return Err(MlError::StringError("MatMul expects 2 inputs".to_string()));
            }
            if input_shapes[0].len() != 2 || input_shapes[1].len() != 2 {
                return Err(MlError::StringError(
                    "MatMul expects 2D inputs".to_string(),
                ));
            }
            let (m, n) = (input_shapes[0][0], input_shapes[0][1]);
            let k = input_shapes[1][1];
            if input_shapes[1][0] != n {
                return Err(MlError::StringError(format!(
                    "MatMul shape mismatch: lhs {:?} rhs {:?}",
                    input_shapes[0], input_shapes[1]
                )));
            }
            backend.matmul(
                &input_buffers[0],
                &input_buffers[1],
                m, n, k,
            )
        }
        Op::Copy => {
            if node.inputs.len() != 1 {
                return Err(MlError::StringError("Copy expects 1 input".to_string()));
            }
            input_buffers[0].clone()
        }
        Op::Sum { .. } | Op::Reshape { .. } => {
            return Err(MlError::StringError(format!(
                "op {:?} not yet supported in graph execution",
                node.op
            )));
        }
    };
    Ok(result)
}

/// Executes a validated graph on the given backend with the provided inputs.
///
/// Use [`crate::tensor::compile_for_execution`] to validate a [`crate::tensor::TensorGraph`] first.
/// `input_data` and `input_shapes` must have length `graph.input_count()`; each
/// `input_data[i]` must have length equal to the product of `input_shapes[i]`.
/// Only F32 dtype is supported. Supported ops: Add, Sub, Mul, Div, MatMul, Copy.
///
/// Returns one buffer per graph node (in node order), each with length equal to
/// the product of that node's output shape.
pub fn execute_graph<G: ExecutableGraph + ?Sized>(
    backend: &dyn Backend,
    graph: &G,
    input_data: &[Vec<f32>],
    input_shapes: &[Vec<usize>],
) -> MlResult<Vec<Vec<f32>>> {
    if graph.input_count() != input_data.len() || graph.input_count() != input_shapes.len() {
        return Err(MlError::StringError(format!(
            "graph has {} inputs but got {} data buffers and {} shape buffers",
            graph.input_count(),
            input_data.len(),
            input_shapes.len()
        )));
    }

    for (i, (data, shape)) in input_data.iter().zip(input_shapes.iter()).enumerate() {
        let expected = shape_num_elements(shape);
        if data.len() != expected {
            return Err(MlError::StringError(format!(
                "input {} length {} does not match shape {:?} ({} elements)",
                i, data.len(), shape, expected
            )));
        }
    }

    let input_count = graph.input_count();
    let mut buffers: Vec<Vec<f32>> = input_data.to_vec();
    let mut shapes: Vec<Vec<usize>> = input_shapes.to_vec();

    for node in graph.nodes() {
        if node.output.dtype_id != DTypeId::F32 {
            return Err(MlError::StringError(
                "only F32 dtype is supported for graph execution".to_string(),
            ));
        }
        let out_len = shape_num_elements(&node.output.shape);
        shapes.push(node.output.shape.clone());
        buffers.push(vec![0.0; out_len]);
    }

    for level in graph.parallel_levels() {
        for node_id in level {
            let node = graph.node(node_id).ok_or_else(|| {
                MlError::StringError("node missing in graph".to_string())
            })?;
            let out_idx = input_count + node_id.0;
            let input_buffers: Vec<Vec<f32>> = node
                .inputs
                .iter()
                .map(|r| buffers[buffer_index(input_count, *r)].clone())
                .collect();
            let input_shapes_for_node: Vec<Vec<usize>> = node
                .inputs
                .iter()
                .map(|r| shapes[buffer_index(input_count, *r)].clone())
                .collect();
            let result =
                execute_one_node(backend, node, &input_buffers, &input_shapes_for_node)?;
            buffers[out_idx].copy_from_slice(&result);
        }
    }

    let node_outputs = buffers.split_off(input_count);
    Ok(node_outputs)
}

/// Like [`execute_graph`], but runs nodes in the same parallel level concurrently using
/// threads. Use this when the backend is CPU and the graph has independent branches
/// (e.g. fan-out from inputs). Requires `backend: Arc<dyn Backend>` so it can be
/// shared across threads.
pub fn execute_graph_parallel<G: ExecutableGraph + ?Sized>(
    backend: Arc<dyn Backend>,
    graph: &G,
    input_data: &[Vec<f32>],
    input_shapes: &[Vec<usize>],
) -> MlResult<Vec<Vec<f32>>> {
    if graph.input_count() != input_data.len() || graph.input_count() != input_shapes.len() {
        return Err(MlError::StringError(format!(
            "graph has {} inputs but got {} data buffers and {} shape buffers",
            graph.input_count(),
            input_data.len(),
            input_shapes.len()
        )));
    }

    for (i, (data, shape)) in input_data.iter().zip(input_shapes.iter()).enumerate() {
        let expected = shape_num_elements(shape);
        if data.len() != expected {
            return Err(MlError::StringError(format!(
                "input {} length {} does not match shape {:?} ({} elements)",
                i, data.len(), shape, expected
            )));
        }
    }

    let input_count = graph.input_count();
    let mut buffers: Vec<Vec<f32>> = input_data.to_vec();
    let mut shapes: Vec<Vec<usize>> = input_shapes.to_vec();

    for node in graph.nodes() {
        if node.output.dtype_id != DTypeId::F32 {
            return Err(MlError::StringError(
                "only F32 dtype is supported for graph execution".to_string(),
            ));
        }
        let out_len = shape_num_elements(&node.output.shape);
        shapes.push(node.output.shape.clone());
        buffers.push(vec![0.0; out_len]);
    }

    for level in graph.parallel_levels() {
        if level.len() <= 1 {
            for node_id in level {
                let node = graph.node(node_id).ok_or_else(|| {
                    MlError::StringError("node missing in graph".to_string())
                })?;
                let out_idx = input_count + node_id.0;
                let input_buffers: Vec<Vec<f32>> = node
                    .inputs
                    .iter()
                    .map(|r| buffers[buffer_index(input_count, *r)].clone())
                    .collect();
                let input_shapes_for_node: Vec<Vec<usize>> = node
                    .inputs
                    .iter()
                    .map(|r| shapes[buffer_index(input_count, *r)].clone())
                    .collect();
                let result = execute_one_node(
                    backend.as_ref(),
                    node,
                    &input_buffers,
                    &input_shapes_for_node,
                )?;
                buffers[out_idx].copy_from_slice(&result);
            }
        } else {
            let results: Vec<(NodeId, MlResult<Vec<f32>>)> = std::thread::scope(|s| {
                let mut handles = Vec::with_capacity(level.len());
                for &node_id in level.iter() {
                    let node = graph.node(node_id).expect("node in level").clone();
                    let input_buffers: Vec<Vec<f32>> = node
                        .inputs
                        .iter()
                        .map(|r| buffers[buffer_index(input_count, *r)].clone())
                        .collect();
                    let input_shapes_for_node: Vec<Vec<usize>> = node
                        .inputs
                        .iter()
                        .map(|r| shapes[buffer_index(input_count, *r)].clone())
                        .collect();
                    let backend_clone = Arc::clone(&backend);
                    handles.push(s.spawn(move || {
                        (
                            node_id,
                            execute_one_node(
                                backend_clone.as_ref(),
                                &node,
                                &input_buffers,
                                &input_shapes_for_node,
                            ),
                        )
                    }));
                }
                handles
                    .into_iter()
                    .map(|h| h.join().expect("thread join"))
                    .collect()
            });
            for (node_id, res) in results {
                let result = res?;
                let out_idx = input_count + node_id.0;
                buffers[out_idx].copy_from_slice(&result);
            }
        }
    }

    let node_outputs = buffers.split_off(input_count);
    Ok(node_outputs)
}

#[cfg(all(test, feature = "cpu"))]
mod tests {
    use super::*;
    use crate::backend::{Device, CpuBackend};
    use crate::tensor::{Op, TensorDesc, TensorGraph, TensorRef};
    use numina::DTypeId;

    #[test]
    fn execute_graph_add_mul() -> MlResult<()> {
        let backend = CpuBackend::new()?;
        let mut graph = TensorGraph::new();
        let a = graph.add_input(vec![2, 2], DTypeId::F32);
        let b = graph.add_input(vec![2, 2], DTypeId::F32);
        let c = graph.add_node(
            Op::Add,
            vec![a, b],
            TensorDesc {
                shape: vec![2, 2],
                dtype_id: DTypeId::F32,
            },
        )?;
        let _d = graph.add_node(
            Op::Mul,
            vec![TensorRef::Node(c), b],
            TensorDesc {
                shape: vec![2, 2],
                dtype_id: DTypeId::F32,
            },
        )?;

        let input_data: Vec<Vec<f32>> = vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![10.0, 20.0, 30.0, 40.0],
        ];
        let input_shapes: Vec<Vec<usize>> = vec![vec![2, 2], vec![2, 2]];

        let outputs = execute_graph(&backend, &graph, &input_data, &input_shapes)?;
        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0], &[11.0, 22.0, 33.0, 44.0]);
        assert_eq!(outputs[1], &[110.0, 440.0, 990.0, 1760.0]);
        Ok(())
    }

    #[test]
    fn execute_graph_parallel_fan_out() -> MlResult<()> {
        let backend = Arc::new(CpuBackend::new()?);
        let mut graph = TensorGraph::new();
        let a = graph.add_input(vec![4], DTypeId::F32);
        let b = graph.add_input(vec![4], DTypeId::F32);
        let _c = graph.add_node(
            Op::Add,
            vec![a, b],
            TensorDesc {
                shape: vec![4],
                dtype_id: DTypeId::F32,
            },
        )?;
        let _d = graph.add_node(
            Op::Sub,
            vec![TensorRef::Input(0), TensorRef::Input(1)],
            TensorDesc {
                shape: vec![4],
                dtype_id: DTypeId::F32,
            },
        )?;
        let c = crate::tensor::NodeId(0);
        let d = crate::tensor::NodeId(1);
        let _e = graph.add_node(
            Op::Mul,
            vec![TensorRef::Node(c), TensorRef::Node(d)],
            TensorDesc {
                shape: vec![4],
                dtype_id: DTypeId::F32,
            },
        )?;

        let input_data: Vec<Vec<f32>> =
            vec![vec![1.0, 2.0, 3.0, 4.0], vec![10.0, 20.0, 30.0, 40.0]];
        let input_shapes: Vec<Vec<usize>> = vec![vec![4], vec![4]];

        let outputs =
            execute_graph_parallel(backend, &graph, &input_data, &input_shapes)?;
        assert_eq!(outputs.len(), 3);
        assert_eq!(outputs[0], &[11.0, 22.0, 33.0, 44.0]);
        assert_eq!(outputs[1], &[-9.0, -18.0, -27.0, -36.0]);
        assert_eq!(outputs[2], &[-99.0, -396.0, -891.0, -1584.0]);
        Ok(())
    }
}
