//!
//! Pin partitions to particular threads on Linux machines.
//! This allows the partition to live in L3 cache for the
//! entire duration of the propagation rather than being
//! shuttled around between cores by rayon's scheduler.
//! 
//! The idea to pin partitions to particular threads 
//! was adopted from monoprop [1].
//!
//! [1]: https://github.com/Algorithmiq/monoprop
//!
//! Copyright and license from [1]:
//!
//! Copyright 2026 Algorithmiq
//!
//! Licensed under the Apache License, Version 2.0 (the "License");
//! you may not use this file except in compliance with the License.
//! You may obtain a copy of the License at
//!
//! <http://www.apache.org/licenses/LICENSE-2.0>
//!
//! Unless required by applicable law or agreed to in writing, software
//! distributed under the License is distributed on an "AS IS" BASIS,
//! WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//! See the License for the specific language governing permissions and
//! limitations under the License.
//!

#[cfg(target_os = "linux")]
pub fn available_cpus() -> Vec<usize> {
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        if libc::sched_getaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mut set) != 0 {
            return Vec::new();
        }
        (0..libc::CPU_SETSIZE as usize)
            .filter(|&c| libc::CPU_ISSET(c, &set))
            .collect()
    }
}

/// Pins the calling thread to `cpu`, returning true if the kernel accepted it.
#[cfg(target_os = "linux")]
pub fn pin_current_thread(cpu: usize) -> bool {
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_ZERO(&mut set);
        libc::CPU_SET(cpu, &mut set);
        libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set) == 0
    }
}

#[cfg(not(target_os = "linux"))]
pub fn available_cpus() -> Vec<usize> {
    Vec::new()
}

#[cfg(not(target_os = "linux"))]
pub fn pin_current_thread(_cpu: usize) -> bool {
    false
}

/// The CPU worker `index` should occupy, given the CPUs on offer.
pub fn cpu_for_worker(index: usize, cpus: &[usize]) -> Option<usize> {
    if cpus.is_empty() {
        return None;
    }
    Some(cpus[index % cpus.len()])
}
