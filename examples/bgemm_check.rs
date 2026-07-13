use cetana::backend::DeviceManager;
use cetana::tensor::Tensor;

fn main() -> cetana::MlResult<()> {
    let dm = DeviceManager::new();
    let dev = dm.select_device(None)?;
    DeviceManager::set_default_device(dev)?;
    println!("device: {dev:?}");

    for (batch, m, k, n) in [
        (6usize, 5usize, 7usize, 4usize),
        (16, 32, 32, 32), // attention qk^T shape at smoke config
        (16, 32, 32, 64),
        (4, 64, 128, 65), // lm-head-ish
    ] {
        let a: Vec<f32> =
            (0..batch * m * k).map(|i| ((i * 37 % 19) as f32 - 9.0) * 0.1).collect();
        let b: Vec<f32> =
            (0..batch * k * n).map(|i| ((i * 53 % 23) as f32 - 11.0) * 0.1).collect();

        // Naive reference
        let mut reference = vec![0.0f32; batch * m * n];
        for bi in 0..batch {
            for i in 0..m {
                for j in 0..n {
                    let mut s = 0.0f32;
                    for l in 0..k {
                        s += a[bi * m * k + i * k + l] * b[bi * k * n + l * n + j];
                    }
                    reference[bi * m * n + i * n + j] = s;
                }
            }
        }

        let ta = Tensor::new_from_vec(a, &[batch, m, k])?;
        let tb = Tensor::new_from_vec(b, &[batch, k, n])?;
        let gpu = cetana::autograd::bmm(&ta, &tb)?;
        let host = ta.matmul(&tb)?;

        let mut gpu_err = 0.0f32;
        let mut host_err = 0.0f32;
        for i in 0..reference.len() {
            gpu_err = gpu_err.max((gpu.data()[i] - reference[i]).abs());
            host_err = host_err.max((host.data()[i] - reference[i]).abs());
        }
        println!("[{batch},{m},{k},{n}] gpu err {gpu_err:.2e} | host err {host_err:.2e}");
        assert!(gpu_err < 1e-3, "BGEMM mismatch");
    }
    // Repeated calls with a warm buffer pool + interleaved same-size elementwise ops,
    // mimicking the training loop's allocation pattern.
    let (batch, m, k, n) = (16usize, 32usize, 32usize, 32usize);
    for round in 0..10 {
        let a: Vec<f32> = (0..batch * m * k)
            .map(|i| (((i + round * 7) * 37 % 19) as f32 - 9.0) * 0.1)
            .collect();
        let b: Vec<f32> = (0..batch * k * n)
            .map(|i| (((i + round * 3) * 53 % 23) as f32 - 11.0) * 0.1)
            .collect();
        let mut reference = vec![0.0f32; batch * m * n];
        for bi in 0..batch {
            for i in 0..m {
                for j in 0..n {
                    let mut s = 0.0f32;
                    for l in 0..k {
                        s += a[bi * m * k + i * k + l] * b[bi * k * n + l * n + j];
                    }
                    reference[bi * m * n + i * n + j] = s;
                }
            }
        }
        let ta = Tensor::new_from_vec(a.clone(), &[batch, m, k])?;
        let tb = Tensor::new_from_vec(b.clone(), &[batch, k, n])?;
        let gpu = cetana::autograd::bmm(&ta, &tb)?;
        let mut errpos = None;
        let mut gpu_err = 0.0f32;
        for i in 0..reference.len() {
            let e = (gpu.data()[i] - reference[i]).abs();
            if e > gpu_err {
                gpu_err = e;
                errpos = Some(i);
            }
        }
        println!("round {round}: err {gpu_err:.2e} at {errpos:?}");
        // interleave same-size elementwise GPU op (goes through the same pool bucket)
        let backend = ta.get_backend();
        let _ = backend.add(&a, &b);
        let _ = backend.multiply(&a, &b);
    }
    println!("BGEMM OK");
    Ok(())
}
