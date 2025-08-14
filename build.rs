extern crate cc;

fn main() {
    cc::Build::new()
        .cuda(true)
        .cpp(true)
        .flag("-cudart=shared")
        .files(&["./src/vec_add.cu"])
        .compile("vec_add.a");
}
