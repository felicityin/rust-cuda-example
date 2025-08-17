use std::ffi::c_float;

use kernels::*;
use rand::Rng;

extern crate rand;

const VEC_SIZE: usize = 10;
const MAX: f32 = 10.;
const MIN: f32 = 0.;

fn main() {
    let mut v1: Vec<f32> = Vec::new();
    let mut v2: Vec<f32> = Vec::new();

    let mut rng = rand::thread_rng();
    for _ in 0..VEC_SIZE {
        v1.push(rng.gen_range(MIN, MAX));
        v2.push(rng.gen_range(MIN, MAX));
    }

    println!("v1: {:?}", v1);
    println!("v2: {:?}", v2);

    let cpu_res: Vec<f32> = v1.iter().zip(v2.iter()).map(|(x, y)| x + y).collect();
    println!("CPU result: {:?}", cpu_res);

    vector_add_v1(&v1, &v2);

    vector_add_v2(&v1, &v2);
}

fn vector_add_v1(v1: &Vec<f32>, v2: &Vec<f32>) {
    let mut gpu_res: Vec<c_float> = vec![0.0f32; VEC_SIZE];
    unsafe {
        launch_vector_add_v1(v1.as_ptr(), v2.as_ptr(), gpu_res.as_mut_ptr(), VEC_SIZE);
    }
    println!("GPU result: {:?}", gpu_res);
}

fn vector_add_v2(v1: &Vec<f32>, v2: &Vec<f32>) {
    let mut gpu_res: Vec<c_float> = vec![0.0f32; VEC_SIZE];
    unsafe {
        launch_vector_add_v2(v1.as_ptr(), v2.as_ptr(), gpu_res.as_mut_ptr(), VEC_SIZE);
    }
    println!("GPU result: {:?}", gpu_res);
}
