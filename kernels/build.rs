use build_kernel::{KernelBuild, KernelType};

fn main() {
    KernelBuild::new(KernelType::Cuda)
        .files([
            "cuda/vec_add.cu",
            "cuda/vec_add_v1.cu",
            "cuda/vec_add_v2.cu",
            "cuda/vec_add_v3.cu",
            "kernels/add_sub.cu",
        ])
        .deps(["cuda", "include"])
        .include("include")
        .include("cuda")
        .compile("kernels");
}
