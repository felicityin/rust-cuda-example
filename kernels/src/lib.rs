use std::ffi::c_float;

unsafe extern "C" {
    pub fn launch_vector_add_v1(
        v1: *const c_float,
        v2: *const c_float,
        result: *mut c_float,
        n: usize,
    );

    pub fn launch_vector_add_v2(
        v1: *const c_float,
        v2: *const c_float,
        result: *mut c_float,
        n: usize,
    );
}
