//! Directed acyclic graph of tensor operations for scheduling and lowering.
//!
//! The graph is a DAG of **nodes** (ops) and **edges** (data dependencies via [`TensorRef`]).
//! Backends use it to run independent ops in parallel: build a [`Graph`], then call
//! [`Graph::parallel_levels`] to get execution waves (nodes in the same level have no
//! data dependency and can be scheduled together).
//!
//! Re-exported as [`TensorGraph`] from the parent module.

use numina::DTypeId;
use std::collections::VecDeque;

use crate::{MlError, MlResult};
use super::TensorError;

/// Identifies a node in the graph (index into the nodes vector).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub usize);

/// Reference to a tensor value in the graph: either a graph input or the output of a node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorRef {
    /// The `i`-th graph input (zero-based).
    Input(usize),
    /// The single output of the given node.
    Node(NodeId),
}

/// Shape and dtype of a tensor in the graph (used for node outputs and validation).
#[derive(Debug, Clone)]
pub struct TensorDesc {
    pub shape: Vec<usize>,
    pub dtype_id: DTypeId,
}

/// Operation kind for a node; parameters (e.g. axes for Sum) are part of the variant.
#[derive(Debug, Clone)]
pub enum Op {
    Add,
    Sub,
    Mul,
    Div,
    MatMul,
    Sum { axes: Vec<usize>, keep_dims: bool },
    Reshape { shape: Vec<usize> },
    Copy,
}

/// A single node in the graph: one op, its input refs, and the output descriptor.
#[derive(Debug, Clone)]
pub struct Node {
    pub op: Op,
    pub inputs: Vec<TensorRef>,
    pub output: TensorDesc,
}

/// Directed acyclic graph of tensor ops. Built by adding inputs then nodes; refs may only
/// point to existing inputs or earlier nodes so the graph stays acyclic and ordered.
#[derive(Debug, Default)]
pub struct Graph {
    pub input_count: usize,
    pub nodes: Vec<Node>,
}

impl Graph {
    /// Creates an empty graph (no inputs, no nodes).
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a graph input with the given shape and dtype; returns a ref to use as a node input.
    pub fn add_input(&mut self, _shape: Vec<usize>, _dtype_id: DTypeId) -> TensorRef {
        let i = self.input_count;
        self.input_count += 1;
        TensorRef::Input(i)
    }

    /// Appends a node that consumes the given tensor refs and produces one output.
    /// Returns an error if any ref is to a non-existent input or to a node index not yet added.
    pub fn add_node(
        &mut self,
        op: Op,
        inputs: Vec<TensorRef>,
        output: TensorDesc,
    ) -> MlResult<NodeId> {
        for (input_index, r) in inputs.iter().enumerate() {
            match r {
                TensorRef::Input(i) => {
                    if *i >= self.input_count {
                        return Err(MlError::TensorError(TensorError::DagInvalidRef {
                            node_id: self.nodes.len(),
                            input_index,
                        }));
                    }
                }
                TensorRef::Node(n) => {
                    if n.0 >= self.nodes.len() {
                        return Err(MlError::TensorError(TensorError::DagInvalidRef {
                            node_id: self.nodes.len(),
                            input_index,
                        }));
                    }
                }
            }
        }
        let id = NodeId(self.nodes.len());
        self.nodes.push(Node {
            op,
            inputs,
            output,
        });
        Ok(id)
    }

    /// Returns the node for the given id, or `None` if out of range.
    pub fn node(&self, id: NodeId) -> Option<&Node> {
        self.nodes.get(id.0)
    }

    /// Returns node ids in topological order (inputs and earlier nodes before dependents).
    /// With the current builder this is always `0..nodes.len()`.
    pub fn topological_order(&self) -> Vec<NodeId> {
        (0..self.nodes.len()).map(NodeId).collect()
    }

    /// Returns execution waves: each inner `Vec<NodeId>` is a set of nodes that can run in
    /// parallel (no data dependency between them); waves are ordered so earlier waves
    /// complete before later ones need their outputs.
    pub fn parallel_levels(&self) -> Vec<Vec<NodeId>> {
        let n = self.nodes.len();
        let mut in_degree = vec![0usize; n];
        let mut adjacency: Vec<Vec<usize>> = vec![vec![]; n];

        for (node_id, node) in self.nodes.iter().enumerate() {
            for r in &node.inputs {
                let pred = match r {
                    TensorRef::Input(_) => continue,
                    TensorRef::Node(p) => p.0,
                };
                adjacency[pred].push(node_id);
                in_degree[node_id] += 1;
            }
        }

        let mut queue = VecDeque::new();
        for (i, &d) in in_degree.iter().enumerate() {
            if d == 0 {
                queue.push_back(i);
            }
        }

        let mut levels = Vec::new();
        while !queue.is_empty() {
            let level_size = queue.len();
            let mut level = Vec::with_capacity(level_size);
            for _ in 0..level_size {
                let u = queue.pop_front().expect("queue length fixed at loop start");
                level.push(NodeId(u));
                for &v in &adjacency[u] {
                    in_degree[v] -= 1;
                    if in_degree[v] == 0 {
                        queue.push_back(v);
                    }
                }
            }
            levels.push(level);
        }
        levels
    }

    /// Returns true if the graph contains a cycle (invalid). With the current builder,
    /// refs only point backward so cycles cannot be constructed.
    pub fn has_cycle(&self) -> bool {
        let order = self.topological_order();
        let n = self.nodes.len();
        let mut seen = vec![false; n];
        for id in order {
            let node = &self.nodes[id.0];
            for r in &node.inputs {
                if let TensorRef::Node(p) = r {
                    if p.0 >= id.0 {
                        return true;
                    }
                    seen[p.0] = true;
                }
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numina::DTypeId;

    #[test]
    fn dag_add_linear_chain() -> MlResult<()> {
        let mut g = Graph::new();
        let a = g.add_input(vec![2, 2], DTypeId::F32);
        let b = g.add_input(vec![2, 2], DTypeId::F32);
        let c = g.add_node(
            Op::Add,
            vec![a, b],
            TensorDesc {
                shape: vec![2, 2],
                dtype_id: DTypeId::F32,
            },
        )?;
        let d = g.add_node(
            Op::Mul,
            vec![TensorRef::Node(c), b],
            TensorDesc {
                shape: vec![2, 2],
                dtype_id: DTypeId::F32,
            },
        )?;
        assert_eq!(g.input_count, 2);
        assert_eq!(g.nodes.len(), 2);
        assert!(!g.has_cycle());
        let levels = g.parallel_levels();
        assert_eq!(levels.len(), 2);
        assert_eq!(levels[0].len(), 1);
        assert_eq!(levels[1].len(), 1);
        let _ = d;
        Ok(())
    }

    #[test]
    fn dag_parallel_fan_in() -> MlResult<()> {
        let mut g = Graph::new();
        let a = g.add_input(vec![4], DTypeId::F32);
        let b = g.add_input(vec![4], DTypeId::F32);
        let c = g.add_node(
            Op::Add,
            vec![a, b],
            TensorDesc {
                shape: vec![4],
                dtype_id: DTypeId::F32,
            },
        )?;
        let d = g.add_node(
            Op::Sub,
            vec![TensorRef::Input(0), TensorRef::Input(1)],
            TensorDesc {
                shape: vec![4],
                dtype_id: DTypeId::F32,
            },
        )?;
        let _e = g.add_node(
            Op::Mul,
            vec![TensorRef::Node(c), TensorRef::Node(d)],
            TensorDesc {
                shape: vec![4],
                dtype_id: DTypeId::F32,
            },
        )?;
        assert_eq!(g.nodes.len(), 3);
        let levels = g.parallel_levels();
        assert_eq!(levels.len(), 2);
        assert_eq!(levels[0].len(), 2);
        assert_eq!(levels[1].len(), 1);
        Ok(())
    }

    #[test]
    fn dag_invalid_ref_rejected() {
        let mut g = Graph::new();
        let _ = g.add_input(vec![2], DTypeId::F32);
        let err = g.add_node(
            Op::Add,
            vec![TensorRef::Node(NodeId(5)), TensorRef::Input(0)],
            TensorDesc {
                shape: vec![2],
                dtype_id: DTypeId::F32,
            },
        );
        assert!(err.is_err());
    }
}
