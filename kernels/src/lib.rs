// Modified by felicityin
// Copyright RISC Zero, Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

pub mod buffer;
pub mod perf;
pub mod tracker;

pub use buffer::*;
pub use perf::*;
pub use tracker::*;

use std::ffi::CStr;
use std::ffi::c_float;

use anyhow::{Result, anyhow};
use cust::memory::DevicePointer;
use cust::prelude::*;

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
    ) -> *const std::os::raw::c_char;

    pub fn launch_vector_add_v3(
        v1: DevicePointer<u8>,
        v2: DevicePointer<u8>,
        result: DevicePointer<u8>,
        n: usize,
    ) -> *const std::os::raw::c_char;

    pub fn add_sub_events_to_rows(
        events: DevicePointer<u8>,
        rows: DevicePointer<u8>,
        n: usize,
    ) -> *const std::os::raw::c_char;
}

pub fn ffi_wrap<F>(mut inner: F) -> Result<()>
where
    F: FnMut() -> *const std::os::raw::c_char,
{
    unsafe extern "C" {
        fn free(str: *const std::os::raw::c_char);
    }

    let c_ptr = inner();
    if c_ptr.is_null() {
        Ok(())
    } else {
        let what = unsafe {
            let msg = CStr::from_ptr(c_ptr)
                .to_str()
                .unwrap_or("Invalid error msg pointer")
                .to_string();
            free(c_ptr);
            msg
        };
        Err(anyhow!(what))
    }
}

pub fn init_cuda() -> Context {
    cust::init(CudaFlags::empty()).unwrap();
    let device = Device::get_device(0).unwrap();
    let context = Context::new(device).unwrap();
    context.set_flags(ContextFlags::SCHED_AUTO).unwrap();
    context
}
