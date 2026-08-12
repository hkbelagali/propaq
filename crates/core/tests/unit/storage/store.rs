use super::*;

#[test]
fn production_term_sum_has_no_persistent_dense_planes() {
    let stride = 64; // 4096 qubits per plane
    let n_rows = 1000;
    let mut terms = TermSum::<f64>::new(4096, stride);
    let mut x = vec![0u64; stride];
    let z = vec![0u64; stride];
    for i in 0..n_rows {
        x[i % stride] = 1;
        terms.push([&x, &z], 1.0);
        x[i % stride] = 0;
    }
    assert_eq!(terms.len(), n_rows);

    let dense_bytes = 2 * stride * n_rows * std::mem::size_of::<u64>();
    assert!(
        terms.sparse_key_bytes() * 8 < dense_bytes,
        "key storage ({} bytes) is not sparse against the dense equivalent ({dense_bytes} bytes)",
        terms.sparse_key_bytes()
    );
}

#[test]
fn decode_row_reproduces_the_pushed_planes() {
    let stride = 3;
    let mut terms = TermSum::<f64>::new(160, stride);
    let x = [0b101u64, 0, 1 << 40];
    let z = [0u64, 1 << 7, 0];
    terms.push([&x, &z], 2.0);
    let mut buf = vec![0u64; 2 * stride];
    let planes = terms.decode_row(0, &mut buf);
    assert_eq!(planes[0], &x[..]);
    assert_eq!(planes[1], &z[..]);
}
