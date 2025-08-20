// Modified by felicityin
// Copyright RISC Zero, Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use std::{
    fmt::Debug,
    sync::{Mutex, OnceLock},
};

pub fn tracker() -> &'static Mutex<MemoryTracker> {
    static ONCE: OnceLock<Mutex<MemoryTracker>> = OnceLock::new();
    ONCE.get_or_init(|| Mutex::new(MemoryTracker::default()))
}

#[derive(Debug, Default)]
pub struct MemoryTracker {
    pub total: isize,
    pub peak: isize,
}

impl MemoryTracker {
    pub fn reset(&mut self) {
        self.total = 0;
        self.peak = 0;
    }

    pub fn alloc(&mut self, size: usize) {
        self.total += size as isize;
        self.peak = self.peak.max(self.total);
    }

    pub fn free(&mut self, size: usize) {
        self.total -= size as isize;
    }
}
