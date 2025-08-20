// Modified by felicityin
// Copyright RISC Zero, Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use core::fmt::Display;

pub use puffin;

#[doc(hidden)]
pub struct NvtxRange;

impl NvtxRange {
    #[doc(hidden)]
    #[inline]
    #[must_use]
    pub fn new<M: Display>(msg: M) -> Self {
        nvtx::__private::_range_push(msg);
        Self
    }
}

impl Drop for NvtxRange {
    #[inline]
    fn drop(&mut self) {
        nvtx::__private::_range_pop();
    }
}

/// Opens a scope.
#[macro_export]
macro_rules! scope {
    ($name:expr) => {
        // Keep range alive until caller's block scope ends.
        let _nvtx = $crate::perf::NvtxRange::new($name);
        $crate::perf::puffin::profile_scope!($name);
    };

    ($name:expr, $body:expr) => {{
        // Keep range alive while `$body` is evaluated.
        let _nvtx = $crate::perf::NvtxRange::new($name);
        $crate::perf::puffin::profile_scope!($name);
        $body
    }};
}

/// Opens a scope with a formatted message.
#[macro_export]
macro_rules! scope_with {
    ($name:expr, $data:expr) => {
        // Keep range alive until caller's block scope ends.
        let _nvtx = $crate::perf::NvtxRange::new(::core::format_args!($name, $data));
        $crate::perf::puffin::profile_scope!($name, $data);
    };

    ($name:expr, $data:expr, $body:expr) => {{
        // Keep range alive while `$body` is evaluated.
        let _nvtx = $crate::perf::NvtxRange::new(::core::format_args!($name, $data));
        $crate::perf::puffin::profile_scope!($name, $data);
        $body
    }};
}
