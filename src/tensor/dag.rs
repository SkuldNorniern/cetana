//! Directed acyclic graph of tensor operations for scheduling and lowering.
//!
//! Built on the generic [`laminax_dag::Dag`] with shared [`Node`], [`Op`], [`TensorDesc`]
//! from laminax-types. Backends use [`parallel_levels`](Graph::parallel_levels) for execution waves.

use laminax_dag::{Dag, DagError};
use numina::DTypeId;

use crate::{MlError, MlResult};
use super::TensorError;

pub use laminax_dag::{NodeId, Ref as TensorRef};
pub use laminax_types::{Node, Op, TensorDesc};

/// Trait for a graph that can be executed by the backend (raw [`Graph`] or validated [`CompiledGraph`]).
pub trait ExecutableGraph {
    fn input_count(&self) -> usize;
    fn nodes(&self) -> &[Node];
    fn parallel_levels(&self) -> Vec<Vec<NodeId>>;
    fn node(&self, id: NodeId) -> Option<&Node>;
}

/// Tensor op DAG. Built by adding inputs then nodes; refs only point to existing inputs or earlier nodes.
#[derive(Debug, Default, Clone)]
pub struct Graph(Dag<Node>);

impl Graph {
    /// Creates an empty graph (no inputs, no nodes).
    pub fn new() -> Self {
        Self(Dag::new())
    }

    /// Registers a graph input with the given shape and dtype; returns a ref to use as a node input.
    pub fn add_input(&mut self, _shape: Vec<usize>, _dtype_id: DTypeId) -> TensorRef {
        self.0.add_input()
    }

    /// Appends a node that consumes the given tensor refs and produces one output.
    /// Returns an error if any ref is to a non-existent input or to a node index not yet added.
    pub fn add_node(
        &mut self,
        op: Op,
        inputs: Vec<TensorRef>,
        output: TensorDesc,
    ) -> MlResult<NodeId> {
        self.0.add_node(Node { op, inputs, output }).map_err(|e| match e {
            DagError::Cycle => MlError::TensorError(TensorError::DagCycle),
            DagError::InvalidRef { node_id, input_index } => {
                MlError::TensorError(TensorError::DagInvalidRef {
                    node_id,
                    input_index,
                })
            }
        })
    }

    /// Returns the node for the given id, or `None` if out of range.
    pub fn node(&self, id: NodeId) -> Option<&Node> {
        self.0.node(id)
    }

    /// Returns node ids in topological order (inputs and earlier nodes before dependents).
    pub fn topological_order(&self) -> Vec<NodeId> {
        self.0.topological_order()
    }

    /// Returns execution waves: each inner `Vec<NodeId>` is a set of nodes that can run in
    /// parallel (no data dependency between them); waves are ordered so earlier waves
    /// complete before later ones need their outputs.
    pub fn parallel_levels(&self) -> Vec<Vec<NodeId>> {
        self.0.parallel_levels()
    }

    /// Returns true if the graph contains a cycle (invalid). With the current builder,
    /// refs only point backward so cycles cannot be constructed.
    pub fn has_cycle(&self) -> bool {
        self.0.has_cycle()
    }
}

impl ExecutableGraph for Graph {
    fn input_count(&self) -> usize {
        self.0.input_count
    }
    fn nodes(&self) -> &[Node] {
        &self.0.nodes
    }
    fn parallel_levels(&self) -> Vec<Vec<NodeId>> {
        self.0.parallel_levels()
    }
    fn node(&self, id: NodeId) -> Option<&Node> {
        self.0.node(id)
    }
}

/// Validated, executable form of a [`Graph`]. Use as input to backend execution;
/// flow is "build graph -> compile_for_execution -> run". No IR lowering is done yet.
#[derive(Debug, Clone)]
pub struct CompiledGraph {
    inner: Graph,
}

impl CompiledGraph {
    pub fn input_count(&self) -> usize {
        self.inner.0.input_count
    }
    pub fn nodes(&self) -> &[Node] {
        &self.inner.0.nodes
    }
    pub fn parallel_levels(&self) -> Vec<Vec<NodeId>> {
        self.inner.parallel_levels()
    }
    pub fn node(&self, id: NodeId) -> Option<&Node> {
        self.inner.node(id)
    }
}

impl ExecutableGraph for CompiledGraph {
    fn input_count(&self) -> usize {
        self.inner.input_count()
    }
    fn nodes(&self) -> &[Node] {
        self.inner.nodes()
    }
    fn parallel_levels(&self) -> Vec<Vec<NodeId>> {
        self.inner.parallel_levels()
    }
    fn node(&self, id: NodeId) -> Option<&Node> {
        self.inner.node(id)
    }
}

/// Validates the graph (no cycles) and returns an executable form for the backend.
/// Run this before [`crate::backend::execute_graph`]; does not lower to any IR yet.
pub fn compile_for_execution(graph: &Graph) -> MlResult<CompiledGraph> {
    if graph.has_cycle() {
        return Err(MlError::TensorError(TensorError::DagCycle));
    }
    Ok(CompiledGraph {
        inner: graph.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert_eq!(g.input_count(), 2);
        assert_eq!(g.nodes().len(), 2);
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
        assert_eq!(g.nodes().len(), 3);
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
