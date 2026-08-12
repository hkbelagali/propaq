use super::*;

fn rows_of(stride: usize, terms: &[[&[u64]; 2]]) -> SparseRows {
    let mut rows = SparseRows::new(stride);
    for t in terms {
        rows.push_planes(*t);
    }
    rows
}

fn decoded(rows: &SparseRows, i: usize) -> (Vec<u64>, Vec<u64>) {
    let mut a = vec![0u64; rows.stride()];
    let mut b = vec![0u64; rows.stride()];
    rows.decode_into(i, [&mut a, &mut b]);
    (a, b)
}

#[test]
fn empty_store_has_one_offset_and_no_positions() {
    let rows = SparseRows::new(2);
    assert_eq!(rows.len(), 0);
    assert!(rows.is_empty());
    assert_eq!(rows.total_positions(), 0);
}

#[test]
fn an_all_zero_row_is_stored_as_an_empty_position_list() {
    let rows = rows_of(2, &[[&[0, 0], &[0, 0]]]);
    assert_eq!(rows.len(), 1);
    assert_eq!(rows.row(0), &[] as &[Position]);
    assert_eq!(decoded(&rows, 0), (vec![0, 0], vec![0, 0]));
}

#[test]
fn multiword_both_plane_rows_round_trip_bit_for_bit() {
    let stride = 3;
    let a: [u64; 3] = [0b1011, 0, 1 << 63];
    let b: [u64; 3] = [1 << 5, u64::MAX, 0];
    let rows = rows_of(stride, &[[&a, &b]]);
    let (x, z) = decoded(&rows, 0);
    assert_eq!(x, a.to_vec());
    assert_eq!(z, b.to_vec());
    let positions = rows.row(0);
    assert!(
        positions.windows(2).all(|w| w[0] < w[1]),
        "positions must be strictly ascending"
    );
    assert_eq!(
        positions.len() as u32,
        a.iter().chain(&b).map(|w| w.count_ones()).sum::<u32>()
    );
}

#[test]
fn positions_encode_plane_word_and_bit() {
    let rows = rows_of(2, &[[&[0, 1 << 3], &[1 << 2, 0]]]);

    assert_eq!(rows.row(0), &[67, 130]);
}

#[test]
fn compact_drops_flagged_out_rows_and_keeps_order() {
    let mut rows = rows_of(1, &[[&[1], &[0]], [&[2], &[0]], [&[4], &[0]], [&[8], &[0]]]);
    let flags = [1u32, 0, 1, 0];
    let index = [0usize, 1, 1, 2];
    rows.compact(4, &flags, &index, 2);
    assert_eq!(rows.len(), 2);
    assert_eq!(decoded(&rows, 0).0, vec![1]);
    assert_eq!(decoded(&rows, 1).0, vec![4]);
}

#[test]
fn rewrite_rows_can_grow_and_shrink_rows() {
    let mut rows = rows_of(1, &[[&[0b1], &[0]], [&[0b110], &[0]]]);
    rows.rewrite_rows(|i, row, out| {
        if i == 0 {
            out.extend_from_slice(row);
            out.push(64); // add a plane-1 position
        }
        // row 1 collapses to empty
    });
    assert_eq!(rows.len(), 2);
    assert_eq!(decoded(&rows, 0), (vec![1], vec![1]));
    assert_eq!(decoded(&rows, 1), (vec![0], vec![0]));
}

#[test]
fn append_selected_appends_one_row_per_flag_in_row_order() {
    let mut rows = rows_of(1, &[[&[1], &[0]], [&[2], &[0]], [&[4], &[0]]]);
    let flags = [1u32, 0, 1];
    rows.append_selected(3, &flags, |_i, row, out| {
        out.extend_from_slice(row);
        out.push(64);
    });
    assert_eq!(rows.len(), 5);
    assert_eq!(decoded(&rows, 3), (vec![1], vec![1]));
    assert_eq!(decoded(&rows, 4), (vec![4], vec![1]));
}

#[test]
fn word_pair_and_splice_are_inverse_on_the_touched_word() {
    let stride = 2;
    let rows = rows_of(stride, &[[&[0b1010, 0b1], &[0b1, 0b11]]]);
    let span = rows.plane_span();
    assert_eq!(row_word_pair(rows.row(0), span, 0), [0b1010, 0b1]);
    assert_eq!(row_word_pair(rows.row(0), span, 1), [0b1, 0b11]);
    let mut out = Vec::new();
    splice_row_word(rows.row(0), span, 0, [0b1010, 0b1], &mut out);
    assert_eq!(out, rows.row(0));
}

#[test]
fn splice_replaces_only_the_named_word() {
    let stride = 2;
    let rows = rows_of(stride, &[[&[0b1010, 0b1], &[0b1, 0b11]]]);
    let span = rows.plane_span();
    let mut out = Vec::new();
    splice_row_word(rows.row(0), span, 0, [0b1, 0b1010], &mut out);
    let mut replaced = SparseRows::new(stride);
    replaced.push_row(&out);
    assert_eq!(decoded(&replaced, 0), (vec![0b1, 0b1], vec![0b1010, 0b11]));
}

#[test]
fn set_helpers_match_naive_reference() {
    let a: Vec<Position> = vec![1, 4, 7, 9];
    let b: Vec<Position> = vec![2, 4, 9, 11];
    assert_eq!(intersection_count(&a, &b), 2);
    let mut sym = Vec::new();
    symmetric_difference_into(&a, &b, &mut sym);
    assert_eq!(sym, vec![1, 2, 7, 11]);
    let shifted: Vec<Position> = b.iter().map(|p| p + 100).collect();
    assert_eq!(shifted_intersection_count(&a, &shifted, 100), 2);
    assert_eq!(shifted_union_count(&a, &shifted, 100), 6);
}
