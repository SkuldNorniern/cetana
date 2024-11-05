use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=shaders/");
    println!("cargo:rerun-if-changed=cuda/");

    // Compile CUDA kernels
    compile_cuda_kernels();

    // Process SPIR-V shaders
    compile_shaders();
}

fn compile_cuda_kernels() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("Failed to get OUT_DIR"));
    let cuda_dir = PathBuf::from("cuda");

    // Ensure CUDA directory exists
    if !cuda_dir.exists() {
        return;
    }

    // Find all .cu files
    if let Ok(entries) = fs::read_dir(&cuda_dir) {
        for entry in entries.filter_map(Result::ok) {
            if let Some(extension) = entry.path().extension() {
                if extension == "cu" {
                    let output_path = out_dir.join(
                        entry
                            .path()
                            .file_name()
                            .expect("Failed to get filename")
                            .to_str()
                            .expect("Failed to convert to string")
                            .replace(".cu", ".ptx"),
                    );

                    // Run nvcc to compile CUDA kernels to PTX
                    let status = std::process::Command::new("nvcc")
                        .args([
                            "--ptx",
                            "-o",
                            output_path
                                .to_str()
                                .expect("Failed to convert path to string"),
                            entry
                                .path()
                                .to_str()
                                .expect("Failed to convert path to string"),
                        ])
                        .status();

                    match status {
                        Ok(exit_status) if exit_status.success() => {
                            println!(
                                "cargo:warning=Successfully compiled CUDA kernel: {:?}",
                                entry.path()
                            );
                        }
                        _ => panic!("Failed to compile CUDA kernel: {:?}", entry.path()),
                    }
                }
            }
        }
    }
}

fn compile_shaders() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("Failed to get OUT_DIR"));
    let shader_dir = PathBuf::from("shaders");

    // Ensure shader directory exists
    if !shader_dir.exists() {
        return;
    }

    // Find all .comp files (compute shaders)
    if let Ok(entries) = fs::read_dir(&shader_dir) {
        for entry in entries.filter_map(Result::ok) {
            if let Some(extension) = entry.path().extension() {
                if extension == "comp" {
                    let output_path = out_dir.join(
                        entry
                            .path()
                            .file_name()
                            .expect("Failed to get filename")
                            .to_str()
                            .expect("Failed to convert to string")
                            .replace(".comp", ".spv"),
                    );

                    // Run glslc to compile shaders to SPIR-V
                    let status = std::process::Command::new("glslc")
                        .args([
                            entry
                                .path()
                                .to_str()
                                .expect("Failed to convert path to string"),
                            "-o",
                            output_path
                                .to_str()
                                .expect("Failed to convert path to string"),
                        ])
                        .status();

                    match status {
                        Ok(exit_status) if exit_status.success() => {
                            println!(
                                "cargo:warning=Successfully compiled shader: {:?}",
                                entry.path()
                            );
                        }
                        _ => panic!("Failed to compile shader: {:?}", entry.path()),
                    }
                }
            }
        }
    }
}
