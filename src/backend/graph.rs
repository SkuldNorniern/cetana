//! Execution of the tensor op graph on a backend.
//!
//! Runs a [`TensorGraph`] using a [`Backend`]'s primitives (add, multiply, matmul, etc.),
//! respecting [`TensorGraph::parallel_levels`] so nodes in the same level can be scheduled
//! together. Currently only f32 and element-wise / matmul ops are supported.

use crate::backend::Backend;
use crate::tensor::{Op, TensorGraph, TensorRef};
use crate::{MlError, MlResult};
use numina::DTypeId;

fn buffer_index(input_count: usize, r: TensorRef) -> usize {
    match r {
        TensorRef::Input(i) => i,
        TensorRef::Node(n) => input_count + n.0,
    }
}

fn shape_num_elements(shape: &[usize]) -> usize {
    shape.iter().product()
}

/// Executes a tensor graph on the given backend with the provided inputs.
///
/// `input_data` and `input_shapes` must have length `graph.input_count`; each
/// `input_data[i]` must have length equal to the product of `input_shapes[i]`.
/// Only F32 dtype is supported. Supported ops: Add, Sub, Mul, Div, MatMul, Copy.
///
/// Returns one buffer per graph node (in node order), each with length equal to
/// the product of that node's output shape.
pub fn execute_graph(
    backend: &dyn Backend,
    graph: &TensorGraph,
    input_data: &[Vec<f32>],
    input_shapes: &[Vec<usize>],
) -> MlResult<Vec<Vec<f32>>> {
    if graph.input_count != input_data.len() || graph.input_count != input_shapes.len() {
        return Err(MlError::StringError(format!(
            "graph has {} inputs but got {} data buffers and {} shape buffers",
            graph.input_count,
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

    let input_count = graph.input_count;
    let mut buffers: Vec<Vec<f32>> = input_data.to_vec();
    let mut shapes: Vec<Vec<usize>> = input_shapes.to_vec();

    for node in &graph.nodes {
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

            let result: Vec<f32> = match &node.op {
                Op::Add => {
                    if node.inputs.len() != 2 {
                        return Err(MlError::StringError("Add expects 2 inputs".to_string()));
                    }
                    let a_idx = buffer_index(input_count, node.inputs[0]);
                    let b_idx = buffer_index(input_count, node.inputs[1]);
                    backend.add(&buffers[a_idx], &buffers[b_idx])
                }
                Op::Sub => {
                    if node.inputs.len() != 2 {
                        return Err(MlError::StringError("Sub expects 2 inputs".to_string()));
                    }
                    let a_idx = buffer_index(input_count, node.inputs[0]);
                    let b_idx = buffer_index(input_count, node.inputs[1]);
                    backend.sub(&buffers[a_idx], &buffers[b_idx])
                }
                Op::Mul => {
                    if node.inputs.len() != 2 {
                        return Err(MlError::StringError("Mul expects 2 inputs".to_string()));
                    }
                    let a_idx = buffer_index(input_count, node.inputs[0]);
                    let b_idx = buffer_index(input_count, node.inputs[1]);
                    backend.multiply(&buffers[a_idx], &buffers[b_idx])
                }
                Op::Div => {
                    if node.inputs.len() != 2 {
                        return Err(MlError::StringError("Div expects 2 inputs".to_string()));
                    }
                    let a_idx = buffer_index(input_count, node.inputs[0]);
                    let b_idx = buffer_index(input_count, node.inputs[1]);
                    backend.div(&buffers[a_idx], &buffers[b_idx])
                }
                Op::MatMul => {
                    if node.inputs.len() != 2 {
                        return Err(MlError::StringError("MatMul expects 2 inputs".to_string()));
                    }
                    let a_idx = buffer_index(input_count, node.inputs[0]);
                    let b_idx = buffer_index(input_count, node.inputs[1]);
                    let a_shape = &shapes[a_idx];
                    let b_shape = &shapes[b_idx];
                    if a_shape.len() != 2 || b_shape.len() != 2 {
                        return Err(MlError::StringError(
                            "MatMul expects 2D inputs".to_string(),
                        ));
                    }
                    let (m, n) = (a_shape[0], a_shape[1]);
                    let k = b_shape[1];
                    if b_shape[0] != n {
                        return Err(MlError::StringError(format!(
                            "MatMul shape mismatch: lhs {:?} rhs {:?}",
                            a_shape, b_shape
                        )));
                    }
                    backend.matmul(
                        &buffers[a_idx],
                        &buffers[b_idx],
                        m, n, k,
                    )
                }
                Op::Copy => {
                    if node.inputs.len() != 1 {
                        return Err(MlError::StringError("Copy expects 1 input".to_string()));
                    }
                    let src_idx = buffer_index(input_count, node.inputs[0]);
                    buffers[src_idx].clone()
                }
                Op::Sum { .. } | Op::Reshape { .. } => {
                    return Err(MlError::StringError(format!(
                        "op {:?} not yet supported in graph execution",
                        node.op
                    )));
                }
            };

            buffers[out_idx].copy_from_slice(&result);
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
}
