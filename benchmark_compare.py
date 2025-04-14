import torch
import tensorflow as tf
import time
import subprocess
import sys
import os
import re
import argparse

# --- PyTorch Benchmarks ---

def benchmark_pytorch_matmul(dim_a, dim_b, dim_c, device_str, num_runs=10):
    print(f"Benchmarking PyTorch MatMul: ({dim_a}x{dim_b}) * ({dim_b}x{dim_c}) on {device_str}")
    try:
        device = torch.device(device_str)
        a = torch.randn(dim_a, dim_b, device=device)
        b = torch.randn(dim_b, dim_c, device=device)
    except Exception as e:
        print(f"Error setting up PyTorch device '{device_str}': {e}")
        return None

    # Warm-up
    _ = torch.matmul(a, b)
    if device.type == 'cuda': torch.cuda.synchronize()

    start_time = time.perf_counter()
    for _ in range(num_runs):
        _ = torch.matmul(a, b)
    if device.type == 'cuda': torch.cuda.synchronize()
    end_time = time.perf_counter()
    return ((end_time - start_time) / num_runs) * 1000

def benchmark_pytorch_ops(dim, device_str, num_runs=10):
    print(f"Benchmarking PyTorch Ops ({dim}x{dim}) on {device_str}")
    results = {}
    try:
        device = torch.device(device_str)
        a = torch.randn(dim, dim, device=device)
        b = torch.randn(dim, dim, device=device)
        # Ensure non-negative tensor for log/sqrt
        a_pos = torch.abs(a) + 1e-6
    except Exception as e:
        print(f"Error setting up PyTorch device '{device_str}': {e}")
        return None

    ops_to_benchmark = {
        "add": (lambda x, y: x + y, a, b),
        "mul": (lambda x, y: x * y, a, b),
        "relu": (lambda x, y: torch.relu(x), a, b), # y is ignored
        "sum": (lambda x, y: torch.sum(x), a, b),     # y is ignored
        "exp": (lambda x, y: torch.exp(x), a, b),     # y is ignored
        "log": (lambda x, y: torch.log(x), a_pos, b), # y is ignored, use a_pos
        "sqrt": (lambda x, y: torch.sqrt(x), a_pos, b),# y is ignored, use a_pos
        "transpose": (lambda x, y: torch.transpose(x, 0, 1), a, b) # y is ignored
    }

    for name, (op_func, tensor_a, tensor_b) in ops_to_benchmark.items():
        print(f"  Running PyTorch {name}...")
        try:
            # Warm-up
            _ = op_func(tensor_a, tensor_b)
            if device.type == 'cuda': torch.cuda.synchronize()

            start_time = time.perf_counter()
            for _ in range(num_runs):
                _ = op_func(tensor_a, tensor_b)
            if device.type == 'cuda': torch.cuda.synchronize()
            end_time = time.perf_counter()
            results[name] = ((end_time - start_time) / num_runs) * 1000
        except Exception as e:
            print(f"  Error benchmarking PyTorch {name}: {e}")
            results[name] = None

    return results

# --- TensorFlow Benchmarks ---

def benchmark_tensorflow_matmul(dim_a, dim_b, dim_c, device_str, num_runs=10):
    print(f"\nBenchmarking TensorFlow MatMul: ({dim_a}x{dim_b}) * ({dim_b}x{dim_c}) on {device_str}")
    tf_device_name = "/GPU:0" if device_str == "gpu" else "/CPU:0"
    print(f"Attempting to use TensorFlow device: {tf_device_name}")
    try:
        with tf.device(tf_device_name):
            a = tf.random.normal([dim_a, dim_b])
            b = tf.random.normal([dim_b, dim_c])
            # Warm-up
            _ = tf.matmul(a, b).numpy()
            start_time = time.perf_counter()
            for _ in range(num_runs):
                _result = tf.matmul(a, b)
            _result.numpy() # Force execution
            end_time = time.perf_counter()
        return ((end_time - start_time) / num_runs) * 1000
    except Exception as e:
        print(f"Error setting up or running on TensorFlow device '{tf_device_name}': {e}")
        gpus = tf.config.list_physical_devices('GPU')
        if device_str == "gpu" and not gpus: print("Note: No physical GPUs detected/configured for TensorFlow.")
        return None

def benchmark_tensorflow_ops(dim, device_str, num_runs=10):
    print(f"\nBenchmarking TensorFlow Ops ({dim}x{dim}) on {device_str}")
    results = {}
    tf_device_name = "/GPU:0" if device_str == "gpu" else "/CPU:0"
    print(f"Attempting to use TensorFlow device: {tf_device_name}")

    try:
        with tf.device(tf_device_name):
            a = tf.random.normal([dim, dim])
            b = tf.random.normal([dim, dim])
            # Ensure non-negative tensor for log/sqrt
            a_pos = tf.abs(a) + 1e-6

            ops_to_benchmark = {
                "add": (lambda x, y: tf.add(x, y), a, b),
                "mul": (lambda x, y: tf.multiply(x, y), a, b),
                "relu": (lambda x, y: tf.nn.relu(x), a, b),     # y is ignored
                "sum": (lambda x, y: tf.reduce_sum(x), a, b),  # y is ignored
                "exp": (lambda x, y: tf.exp(x), a, b),      # y is ignored
                "log": (lambda x, y: tf.math.log(x), a_pos, b),# y is ignored, use a_pos
                "sqrt": (lambda x, y: tf.sqrt(x), a_pos, b),   # y is ignored, use a_pos
                "transpose": (lambda x, y: tf.transpose(x, perm=[1, 0]), a, b) # y is ignored
            }

            for name, (op_func, tensor_a, tensor_b) in ops_to_benchmark.items():
                print(f"  Running TensorFlow {name}...")
                try:
                    # Warm-up
                    _ = op_func(tensor_a, tensor_b).numpy()
                    start_time = time.perf_counter()
                    for _ in range(num_runs):
                        _result = op_func(tensor_a, tensor_b)
                    _result.numpy() # Force execution
                    end_time = time.perf_counter()
                    results[name] = ((end_time - start_time) / num_runs) * 1000
                except Exception as e:
                     print(f"  Error benchmarking TensorFlow {name}: {e}")
                     results[name] = None
        return results

    except Exception as e:
        print(f"Error setting up or running on TensorFlow device '{tf_device_name}': {e}")
        gpus = tf.config.list_physical_devices('GPU')
        if device_str == "gpu" and not gpus: print("Note: No physical GPUs detected/configured for TensorFlow.")
        return None

# --- Cetana Benchmarks ---

def run_cetana_matmul_benchmark(dim_a, dim_b, dim_c, rust_executable_path):
    print(f"\nRunning Cetana MatMul benchmark: {rust_executable_path}")
    cmd = [rust_executable_path, str(dim_a), str(dim_b), str(dim_c)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=60) # Added timeout
        output = result.stdout
        print(output) # Print the output from the rust program
        match = re.search(r"cetana_ms:(\d+\.?\d*)", output)
        if match: return float(match.group(1))
        else:
            print("Error: Could not parse Cetana MatMul benchmark output.")
            print("STDOUT:", result.stdout); print("STDERR:", result.stderr)
            return None
    except FileNotFoundError:
        print(f"Error: Rust executable not found at {rust_executable_path}")
        print("Did cargo build succeed?")
        return None
    except subprocess.TimeoutExpired:
        print(f"Error: Cetana MatMul benchmark timed out.")
        return None
    except subprocess.CalledProcessError as e:
        print(f"Error running Cetana MatMul benchmark: {e}")
        print("STDOUT:", e.stdout); print("STDERR:", e.stderr)
        return None

def run_cetana_ops_benchmark(op_dim, rust_executable_path):
    print(f"\nRunning Cetana Ops benchmark ({op_dim}x{op_dim}): {rust_executable_path}")
    cmd = [rust_executable_path, str(op_dim)]
    results = {}
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=60)
        output = result.stdout
        print(output)

        # Parse output for each operation
        ops = ["add", "mul", "relu", "sum", "exp", "log", "sqrt", "transpose"] # Added new ops
        for op in ops:
            match = re.search(rf"cetana_{op}_ms:(\d+\.?\d*)", output)
            if match:
                results[op] = float(match.group(1))
            else:
                print(f"Warning: Could not parse Cetana {op} benchmark output.")
                results[op] = None
        return results
    except FileNotFoundError:
        print(f"Error: Rust executable not found at {rust_executable_path}")
        print("Did cargo build succeed?")
        return None
    except subprocess.TimeoutExpired:
        print(f"Error: Cetana Ops benchmark timed out.")
        return None
    except subprocess.CalledProcessError as e:
        print(f"Error running Cetana Ops benchmark: {e}")
        print("STDOUT:", e.stdout); print("STDERR:", e.stderr)
        return None

def compile_rust_benchmark(example_name, feature_flag):
    # feature_flag now comes from the determined effective device
    print(f"\nCompiling Rust benchmark example: {example_name} with feature '{feature_flag}'...")
    project_root = os.path.dirname(os.path.abspath(__file__))
    # print(f"Detected project root: {project_root}")
    cmd = [
        "cargo", "build", "--release", "--example", example_name,
        "--no-default-features",
        "--features", feature_flag
    ]
    print(f"Running compile command: {' '.join(cmd)}")
    try:
        process = subprocess.run(cmd, cwd=project_root, capture_output=True, text=True, check=True, timeout=180)
        # Optionally hide verbose stdout/stderr unless error occurs
        # print("Cargo build stdout:", process.stdout)
        # print("Cargo build stderr:", process.stderr)

        executable_path = os.path.join(project_root, "target", "release", "examples", example_name)
        if sys.platform == "win32": executable_path += ".exe"

        if os.path.exists(executable_path):
            print(f"Compilation successful. Executable at: {executable_path}")
            return executable_path
        else:
            print(f"Error: Compiled executable not found at expected path: {executable_path}")
            print("Check cargo build output. Feature flags might change output location.")
            return None
    except FileNotFoundError:
        print("Error: 'cargo' command not found. Is Rust installed and in PATH?")
        return None
    except subprocess.TimeoutExpired:
         print(f"Error: Compilation of {example_name} with feature {feature_flag} timed out.")
         return None
    except subprocess.CalledProcessError as e:
        print(f"Error during compilation of {example_name} with feature {feature_flag}: {e}")
        print("STDOUT:", e.stdout); print("STDERR:", e.stderr)
        return None

# --- Main Execution ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Benchmark operations across Cetana, PyTorch, and TensorFlow.')
    parser.add_argument('dims', metavar='DIM', type=int, nargs='*', help='Dimensions for MatMul (dim_a dim_b dim_c). Defaults to 1024 1024 1024.')
    parser.add_argument('--op-dim', type=int, default=1024, help='Dimension size (NxN) for element-wise Ops benchmarks. Default: 1024')
    # Single device flag controls PT, TF, and Cetana compilation feature
    parser.add_argument('--device', type=str, default='auto', 
                        choices=['cpu', 'cuda', 'mps', 'rocm', 'auto'], 
                        help='Target device/backend for ALL frameworks (cpu, cuda, mps, rocm, auto). Default: auto')
    # Removed --cetana-feature
    parser.add_argument('--num-runs', type=int, default=10, help='Number of runs for averaging performance. Default: 10')

    args = parser.parse_args()

    # --- Setup --- 
    if len(args.dims) == 3:
        dim_a, dim_b, dim_c = args.dims
    elif len(args.dims) == 0:
        dim_a, dim_b, dim_c = 1024, 1024, 1024
        print(f"Using default MatMul dimensions: {dim_a} {dim_b} {dim_c}")
    else:
        parser.error("Provide exactly three MatMul dimensions (dim_a dim_b dim_c) or none for defaults.")

    op_dim = args.op_dim
    num_runs = args.num_runs
    requested_device = args.device

    # --- Determine Effective Devices and Cetana Compile Feature ---
    print(f"\nRequested device: {requested_device.upper()}")

    # 1. Determine the feature flag for Cetana compilation based on request/auto
    cetana_feature_to_compile = "cpu" # Default for auto initially
    if requested_device == 'auto':
        if torch.cuda.is_available():
            # Prefer CUDA/ROCm if available for auto
            cetana_feature_to_compile = "cuda"
            print("Info: 'auto' detected CUDA/ROCm available via PyTorch. Will compile Cetana with 'cuda' feature.")
        else:
            try:
                if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                    # Use MPS if CUDA not found but MPS is
                    cetana_feature_to_compile = "mps"
                    print("Info: 'auto' detected MPS available via PyTorch. Will compile Cetana with 'mps' feature.")
                else:
                    print("Info: 'auto' detected no GPU/MPS via PyTorch. Will compile Cetana with 'cpu' feature.")
            except AttributeError:
                 print("Info: 'auto' detected no GPU/MPS via PyTorch (MPS backend not found). Will compile Cetana with 'cpu' feature.")
    else:
        # User specified a device, use that for Cetana feature
        cetana_feature_to_compile = requested_device
        # Note: We still need runtime checks for PT/TF below

    # 2. Determine runtime devices for PyTorch and TensorFlow based on request AND availability
    pytorch_device_str = "cpu" # Default
    tensorflow_device_str = "cpu" # Default
    tf_maps_to = "CPU:0" # Default TF device name

    # PyTorch Runtime Check
    if requested_device == 'cuda' or requested_device == 'rocm':
        if torch.cuda.is_available():
            pytorch_device_str = "cuda"
        else:
            print(f"Warning: Device '{requested_device}' requested, but PyTorch CUDA/ROCm unavailable at runtime. Using CPU for PyTorch.")
    elif requested_device == 'mps':
        try:
             if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                 pytorch_device_str = "mps"
             else:
                  print("Warning: Device 'mps' requested, but PyTorch MPS unavailable/not built at runtime. Using CPU for PyTorch.")
        except AttributeError:
             print("Warning: Device 'mps' requested, but PyTorch MPS backend not found in this build. Using CPU for PyTorch.")
    elif requested_device == 'cpu':
        pytorch_device_str = "cpu"
    elif requested_device == 'auto': # Re-check availability for runtime
        if torch.cuda.is_available(): pytorch_device_str = "cuda"
        else:
            try:
                if torch.backends.mps.is_available() and torch.backends.mps.is_built(): pytorch_device_str = "mps"
            except AttributeError: pass # Stays CPU

    # TensorFlow Runtime Check
    tf_gpus = tf.config.list_physical_devices('GPU')
    if requested_device == 'cuda' or requested_device == 'rocm':
        if tf_gpus:
            tensorflow_device_str = "gpu" # TF uses 'gpu' for both
            tf_maps_to = "GPU:0"
        else:
            print(f"Warning: Device '{requested_device}' requested, but TensorFlow GPU (CUDA/ROCm) unavailable/unconfigured at runtime. Using CPU for TensorFlow.")
            # Keep tensorflow_device_str = "cpu"
    elif requested_device == 'mps':
        print("Warning: Device 'mps' requested for TensorFlow. Requires 'tensorflow-metal' plugin (cannot detect). Assuming CPU for TensorFlow runtime.")
        # Keep tensorflow_device_str = "cpu"
    elif requested_device == 'cpu':
        tensorflow_device_str = "cpu"
        # tf_maps_to is already "CPU:0"
    elif requested_device == 'auto': # Re-check availability for runtime
        if tf_gpus:
            tensorflow_device_str = "gpu"
            tf_maps_to = "GPU:0"
    # Else: Keep defaults (cpu)

    # 3. Final Config Output
    print(f"\nFinal Configuration:")
    print(f"  Cetana Compiled Feature: {cetana_feature_to_compile.upper()}")
    print(f"  PyTorch Runtime Device:  {pytorch_device_str.upper()}")
    print(f"  TensorFlow Runtime Device: {tensorflow_device_str.upper()} (as {tf_maps_to})")

    # --- Compile Cetana --- 
    # Pass the feature flag determined above
    cetana_matmul_exec = compile_rust_benchmark("benchmark_matmul", cetana_feature_to_compile)
    cetana_ops_exec = compile_rust_benchmark("benchmark_ops", cetana_feature_to_compile)

    # --- Run Benchmarks --- 
    print("\nStarting benchmarks...")
    # MatMul
    pytorch_matmul_time = benchmark_pytorch_matmul(dim_a, dim_b, dim_c, pytorch_device_str, num_runs)
    tf_matmul_time = benchmark_tensorflow_matmul(dim_a, dim_b, dim_c, tensorflow_device_str, num_runs)
    cetana_matmul_time = None
    if cetana_matmul_exec:
        # This runs the Cetana version compiled with cetana_feature_to_compile
        cetana_matmul_time = run_cetana_matmul_benchmark(dim_a, dim_b, dim_c, cetana_matmul_exec)

    # Other Ops
    print(f"\n--- Benchmarking Element-wise Ops ({op_dim}x{op_dim}) ---")
    pytorch_ops_times = benchmark_pytorch_ops(op_dim, pytorch_device_str, num_runs)
    tf_ops_times = benchmark_tensorflow_ops(op_dim, tensorflow_device_str, num_runs)
    cetana_ops_times = None
    if cetana_ops_exec:
        cetana_ops_times = run_cetana_ops_benchmark(op_dim, cetana_ops_exec)

    # --- Print Results --- 
    print("\n--- MatMul Benchmark Results (Average ms per run) ---")
    # Use the feature Cetana was compiled with in the label
    print(f"Cetana ({cetana_feature_to_compile.upper()}):    {cetana_matmul_time:.4f} ms" if cetana_matmul_time is not None else f"Cetana ({cetana_feature_to_compile.upper()}):    Failed/Not Run")
    print(f"PyTorch ({pytorch_device_str.upper()}):   {pytorch_matmul_time:.4f} ms" if pytorch_matmul_time is not None else f"PyTorch ({pytorch_device_str.upper()}):   Failed/Not Run")
    print(f"TensorFlow ({tensorflow_device_str.upper()}): {tf_matmul_time:.4f} ms" if tf_matmul_time is not None else f"TensorFlow ({tensorflow_device_str.upper()}): Failed/Not Run")

    print(f"\n--- Ops Benchmark Results [{op_dim}x{op_dim}] (Average ms per run) ---")
    ops = ["add", "mul", "relu", "sum", "exp", "log", "sqrt", "transpose"]
    # Use the feature Cetana was compiled with in the header
    header = f"Operation   | Cetana ({cetana_feature_to_compile.upper()})   | PyTorch ({pytorch_device_str.upper()}) | TensorFlow ({tensorflow_device_str.upper()})"
    print(header)
    print("-" * len(header))
    for op in ops:
        c_time = cetana_ops_times.get(op) if cetana_ops_times else None
        p_time = pytorch_ops_times.get(op) if pytorch_ops_times else None
        t_time = tf_ops_times.get(op) if tf_ops_times else None

        c_str = f"{c_time:>13.4f}" if c_time is not None else "     Failed/NR"
        p_str = f"{p_time:>13.4f}" if p_time is not None else "     Failed/NR"
        t_str = f"{t_time:>13.4f}" if t_time is not None else "     Failed/NR"

        print(f"{op:<11} | {c_str} | {p_str} | {t_str}") 