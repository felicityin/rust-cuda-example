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

    vector_add_v3(&v1, &v2);
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
    ffi_wrap(|| unsafe {
        launch_vector_add_v2(v1.as_ptr(), v2.as_ptr(), gpu_res.as_mut_ptr(), VEC_SIZE)
    })
    .unwrap();
    println!("GPU result: {:?}", gpu_res);
}

fn vector_add_v3(v1: &Vec<f32>, v2: &Vec<f32>) {
    let d_v1 = CudaBuffer::<f32>::copy_from("v1", v1);
    let d_v2 = CudaBuffer::<f32>::copy_from("v2", v2);
    let d_res = CudaBuffer::<f32>::new("result", v2.len());

    ffi_wrap(|| unsafe {
        launch_vector_add_v3(
            d_v1.as_device_ptr(),
            d_v2.as_device_ptr(),
            d_res.as_device_ptr(),
            VEC_SIZE,
        )
    })
    .unwrap();
    println!("GPU result: {:?}", d_res.to_vec());
}
