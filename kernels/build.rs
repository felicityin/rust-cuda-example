use build_kernel::{KernelBuild, KernelType};

fn main() {
    println!("cargo:rustc-link-search=/usr/local/cuda-12.4/lib64");
    println!("cargo:rustc-link-lib=cuda");

    KernelBuild::new(KernelType::Cuda)
        .files([
            "kernels/vec_add.cu",
            "kernels/vec_add_v1.cu",
            "kernels/vec_add_v2.cu",
            "kernels/vec_add_v3.cu",
            "kernels/add_sub_chip.cu",
        ])
        .deps(["kernels", "include"])
        .include("include")
        .include("kernels")
        .compile("kernels");
}
