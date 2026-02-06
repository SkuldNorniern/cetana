/// Generic buffer for device memory.
///
/// This trait now uses an associated error type which allows each backend
/// to plug its own error domain while keeping a uniform interface.
#[allow(dead_code)]
pub trait Buffer: std::fmt::Debug {
    type Error: std::error::Error;

    /// Allocates a new buffer on the device.
    fn new(size: usize) -> Result<Self, Self::Error>
    where
        Self: Sized;

    /// Copies data from host to device.
    fn copy_from_host<T: Copy>(&mut self, data: &[T]) -> Result<(), Self::Error>;

    /// Copies data from device to host.
    fn copy_to_host<T: Copy>(&self, data: &mut [T]) -> Result<(), Self::Error>;

    /// Returns the size of the underlying buffer in bytes.
    fn size(&self) -> usize;
}
