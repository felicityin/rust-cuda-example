use build_kernel::{KernelBuild, KernelType};

fn main() {
    KernelBuild::new(KernelType::Cuda)
        .files([
            "cuda/vec_add.cu",
            "cuda/vec_add_v1.cu",
            "cuda/vec_add_v2.cu",
            "cuda/vec_add_v3.cu",
        ])
        .deps(["cuda"])
        .include("cuda")
        .compile("kernels");
}
