use tracing_subscriber::fmt;

mod vector_add;
mod zkm_trace;

fn main() {
    fmt::init();

    vector_add::test_vector_add();

    zkm_trace::test_generate_trace_cuda_eq_cpu();
}
