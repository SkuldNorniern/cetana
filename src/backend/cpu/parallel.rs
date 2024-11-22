use std::sync::Arc;
use std::thread;

#[derive(Debug)]
pub struct ParallelExecutor {
    thread_count: usize,
}

#[cfg(not(feature = "rayon"))]
impl ParallelExecutor {

    pub fn new() -> Self {
        let thread_count = thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);

        ParallelExecutor { thread_count }
    }

    pub fn execute<T, F>(&self, data: &[T], chunk_size: usize, f: F) -> Vec<T>
    where
        T: Send + Sync + Copy + 'static,
        F: Fn(&[T]) -> Vec<T> + Send + Sync + Clone + 'static,
    {
        if data.len() <= chunk_size {
            return f(data);
        }

        let data = Arc::new(data.to_vec());
        let total_elements = data.len();
        let elements_per_thread = (total_elements + self.thread_count - 1) / self.thread_count;
        let mut handles = Vec::with_capacity(self.thread_count);

        for thread_idx in 0..self.thread_count {
            let start = thread_idx * elements_per_thread;
            if start >= total_elements {
                break;
            }

            let end = (start + elements_per_thread).min(total_elements);
            let thread_data = Arc::clone(&data);
            let thread_f = f.clone();

            handles.push(thread::spawn(move || thread_f(&thread_data[start..end])));
        }

        handles
            .into_iter()
            .flat_map(|h| h.join().unwrap())
            .collect()
    }

    pub fn execute_binary<T, F>(&self, a: &[T], b: &[T], chunk_size: usize, f: F) -> Vec<T>
    where
        T: Send + Sync + Copy + 'static,
        F: Fn(&[T], &[T]) -> Vec<T> + Send + Sync + Clone + 'static,
    {
        if a.len() <= chunk_size {
            return f(a, b);
        }

        let a = Arc::new(a.to_vec());
        let b = Arc::new(b.to_vec());
        let total_elements = a.len();
        let elements_per_thread = (total_elements + self.thread_count - 1) / self.thread_count;
        let mut handles = Vec::with_capacity(self.thread_count);

        for thread_idx in 0..self.thread_count {
            let start = thread_idx * elements_per_thread;
            if start >= total_elements {
                break;
            }

            let end = (start + elements_per_thread).min(total_elements);
            let thread_a = Arc::clone(&a);
            let thread_b = Arc::clone(&b);
            let thread_f = f.clone();

            handles.push(thread::spawn(move || {
                thread_f(&thread_a[start..end], &thread_b[start..end])
            }));
        }

        handles
            .into_iter()
            .flat_map(|h| h.join().unwrap())
            .collect()
    }
}

#[cfg(feature = "rayon")]
use rayon::prelude::*;

#[cfg(feature = "rayon")]
impl ParallelExecutor {
    pub fn new() -> Self {
        let thread_count = thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);

        ParallelExecutor { thread_count }
    }

    pub fn execute<T, F>(&self, data: &[T], chunk_size: usize, f: F) -> Vec<T>
    where
        T: Send + Sync + Copy + 'static,
        F: Fn(&[T]) -> Vec<T> + Send + Sync + Clone + 'static,
    {
        if data.len() <= chunk_size {
            return f(data);
        }

        data.par_chunks(chunk_size)
            .flat_map(|chunk| f(chunk))
            .collect()
    }

    pub fn execute_binary<T, F>(&self, a: &[T], b: &[T], chunk_size: usize, f: F) -> Vec<T>
    where
        T: Send + Sync + Copy + 'static,
        F: Fn(&[T], &[T]) -> Vec<T> + Send + Sync + Clone + 'static,
    {
        if a.len() <= chunk_size {
            return f(a, b);
        }

        a.par_chunks(chunk_size)
            .zip(b.par_chunks(chunk_size))
            .flat_map(|(chunk_a, chunk_b)| f(chunk_a, chunk_b))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parallel_execute() {
        let executor = ParallelExecutor::new();
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];

        let result = executor.execute(&data, 2, |chunk| chunk.iter().map(|&x| x * 2.0).collect());

        assert_eq!(result, vec![2.0, 4.0, 6.0, 8.0, 10.0]);
    }

    #[test]
    fn test_parallel_execute_binary() {
        let executor = ParallelExecutor::new();
        let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let b = vec![2.0f32, 3.0, 4.0, 5.0, 6.0];

        let result = executor.execute_binary(&a, &b, 2, |chunk_a, chunk_b| {
            chunk_a
                .iter()
                .zip(chunk_b.iter())
                .map(|(&x, &y)| x + y)
                .collect()
        });

        assert_eq!(result, vec![3.0, 5.0, 7.0, 9.0, 11.0]);
    }
}
