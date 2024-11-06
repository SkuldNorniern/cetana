use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn find_cuda_path() -> String {
    // Linux
    if let Ok(output) = Command::new("which").arg("nvcc").output() {
        if let Ok(path) = String::from_utf8(output.stdout) {
            if let Some(cuda_path) = path.trim().strip_suffix("/bin/nvcc") {
                return cuda_path.to_string();
            }
        }
    }

    // Windows
    for path in &[
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA",
        "C:/CUDA",
    ] {
        if PathBuf::from(path).exists() {
            return path.to_string();
        }
    }

    "/usr/local/cuda".to_string()
}

fn compile_shaders() -> std::io::Result<()> {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    // Compile reduction shader
    println!("cargo:rerun-if-changed=shaders/reduction.comp");
    let status = Command::new("glslc")
        .args([
            "shaders/reduction.comp",
            "-o",
            out_dir.join("reduction.spv").to_str().unwrap(),
        ])
        .status()
        .expect("Failed to execute glslc");

    if !status.success() {
        panic!("Failed to compile reduction shader");
    }

    // Compile binary operations shader
    println!("cargo:rerun-if-changed=shaders/binary_ops.comp");
    let status = Command::new("glslc")
        .args([
            "shaders/binary_ops.comp",
            "-o",
            out_dir.join("binary_ops.spv").to_str().unwrap(),
        ])
        .status()
        .expect("Failed to execute glslc");

    if status.success() {
        Ok(())
    } else {
        Err(std::io::Error::new(std::io::ErrorKind::Other, "Failed to compile binary operations shader"))
    }
}


fn main() {
    println!("cargo:rerun-if-changed=cuda/");
    println!("cargo:rerun-if-changed=cuda-headers/");
    println!("cargo:rerun-if-changed=CMakeLists.txt");

    let cuda_path = find_cuda_path();
    let clangd_path = PathBuf::from(".clangd");

    if !clangd_path.exists() {
        let clangd_content = format!(
            r#"CompileFlags:
  Remove: 
    - "-forward-unknown-to-host-compiler"
    - "-rdc=*"
    - "-Xcompiler*"
    - "--options-file"
    - "--generate-code*"
  Add: 
    - "-xcuda"
    - "-std=c++14"
    - "-I{}/include"
    - "-I../../cuda-headers"
    - "--cuda-gpu-arch=sm_75"
  Compiler: clang

Index:
  Background: Build

Diagnostics:
  UnusedIncludes: None"#,
            cuda_path
        );

        fs::write(".clangd", clangd_content).expect("Failed to write .clangd file");
    }

    // Compile shaders
    compile_shaders().expect("Failed to compile shaders");

    let dst = cmake::Config::new(".")
        .define("CMAKE_BUILD_TYPE", "Release")
        .define("CUDA_PATH", cuda_path.clone())
        .no_build_target(true)
        .build();

    // Search paths - include both lib and lib64
    println!("cargo:rustc-link-search={}/build/lib", dst.display());
    println!("cargo:rustc-link-search={}/build", dst.display());
    println!("cargo:rustc-link-search=native={}/lib", dst.display());
    println!("cargo:rustc-link-search=native={}/lib64", cuda_path.clone());
    println!("cargo:rustc-link-search=native={}/lib", cuda_path.clone());

    // CUDA runtime linking - only essential libraries
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=cuda");

    // Static libraries - if they exist
    if PathBuf::from(format!("{}/build/lib/libnn_ops.a", dst.display())).exists() {
        println!("cargo:rustc-link-arg=-Wl,--whole-archive");
        println!("cargo:rustc-link-lib=static=nn_ops");
        println!("cargo:rustc-link-lib=static=tensor_ops");
        println!("cargo:rustc-link-arg=-Wl,--no-whole-archive");
    }
}
