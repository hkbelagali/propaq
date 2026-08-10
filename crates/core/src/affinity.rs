///
/// Pinning pool workers to cores, so a partition stays on one core for the run.
///
/// The partitioned engine gives every partition a single writer, and a
/// partition's store is small enough to live in one core's private cache. That
/// only pays if the same core keeps serving the same partition: a partition
/// that migrates has to pull its whole store back from L3 or DRAM on the next
/// gate. Rayon's scheduler is free to move work between workers, and the
/// kernel is free to move workers between cores, so both halves of the binding
/// have to be nailed down. This module is the second half; the first is the
/// broadcast-based dispatch in `partitioned.rs`, which sends partition `i` to
/// worker `i`.
///
/// Linux only. Everywhere else the calls below are no-ops and the engine keeps
/// working, just without the locality.
///

/// CPUs this process is allowed to run on, in ascending order.
///
/// Reads the current affinity mask rather than the core count, so a run under
/// `taskset`, `numactl` or a cgroup uses the cores it was actually given
/// instead of colliding on the ones it was not.
#[cfg(target_os = "linux")]
pub fn available_cpus() -> Vec<usize> {
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        if libc::sched_getaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mut set) != 0 {
            return Vec::new();
        }
        (0..libc::CPU_SETSIZE as usize).filter(|&c| libc::CPU_ISSET(c, &set)).collect()
    }
}

/// Pins the calling thread to `cpu`, returning true if the kernel accepted it.
///
/// Failure is not an error worth propagating: an unpinned worker is slower, not
/// wrong, and the engine has to run on hosts where pinning is not permitted.
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
///
/// Packs workers onto consecutive CPUs in the order the affinity mask lists
/// them, which on a multi-socket host fills one socket before touching the
/// next. The exchange writes across partitions on every gate, so keeping the
/// pool inside one socket keeps that traffic off the inter-socket link for as
/// long as the worker count allows.
pub fn cpu_for_worker(index: usize, cpus: &[usize]) -> Option<usize> {
    if cpus.is_empty() {
        return None;
    }
    Some(cpus[index % cpus.len()])
}

