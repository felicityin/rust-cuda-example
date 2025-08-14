use std::ffi::c_float;

use rand::Rng;

extern crate rand;

unsafe extern "C" {
    pub fn launch_vector_add(
        v1: *const c_float,
        v2: *const c_float,
        result: *mut c_float,
        n: usize,
    );
}

const VEC_SIZE: usize = 10;
const MAX: f32 = 10.;
const MIN: f32 = 0.;

fn main() {
    let mut v1: Vec<f32> = Vec::new();
    let mut v2: Vec<f32> = Vec::new();
    let mut gpu_res: Vec<c_float> = vec![0.0f32; VEC_SIZE];

    let mut rng = rand::thread_rng();
    for _ in 0..VEC_SIZE {
        v1.push(rng.gen_range(MIN, MAX));
        v2.push(rng.gen_range(MIN, MAX));
    }

    println!("v1: {:?}", v1);
    println!("v2: {:?}", v2);

    unsafe {
        launch_vector_add(v1.as_ptr(), v2.as_ptr(), gpu_res.as_mut_ptr(), VEC_SIZE);
    }

    println!("GPU result: {:?}", gpu_res);

    let cpu_res: Vec<f32> = v1.iter().zip(v2.iter()).map(|(x, y)| x + y).collect();
    println!("CPU result: {:?}", cpu_res);
}
