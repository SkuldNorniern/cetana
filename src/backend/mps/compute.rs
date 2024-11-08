use super::core::MpsDevice;
use metal::{CommandQueue, ComputePipelineState, MTLSize};
use std::sync::Arc;

pub struct MpsCompute {
    device: Arc<MpsDevice>,
    command_queue: CommandQueue,
}

impl MpsCompute {
    pub fn new(device: Arc<MpsDevice>) -> Result<Self, crate::backend::MpsError> {
        let command_queue = device
            .device()
            .new_command_queue();
            // Read comments in src/backend/mps/backend.rs create_buffer for more information
            // .ok_or(crate::MpsError::InitializationError)?;

        Ok(Self {
            device,
            command_queue,
        })
    }

    pub fn dispatch_compute(
        &self,
        pipeline: &ComputePipelineState,
        grid_size: MTLSize,
        thread_group_size: MTLSize,
    ) -> Result<(), crate::backend::MpsError> {
        let command_buffer = self.command_queue.new_command_buffer();
        let compute_encoder = command_buffer.new_compute_command_encoder();

        compute_encoder.set_compute_pipeline_state(pipeline);
        compute_encoder.dispatch_thread_groups(grid_size, thread_group_size);
        compute_encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(())
    }
}

impl Drop for MpsCompute {
    fn drop(&mut self) {
        // Metal handles cleanup automatically through reference counting
    }
}
