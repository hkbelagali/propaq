///
/// Some helper stuff for the core crate to convert 
// between Python integers and Bitsets. 
///
use pyo3::prelude::*;
use crate::bitset::Bitset;

pub fn pyint_to_bitset(obj: &Bound<'_, PyAny>, _n_bits: usize) -> PyResult<Bitset> {
    let bit_length: usize = obj.call_method0("bit_length")?.extract()?;
    let byte_len = (bit_length + 7) / 8;
    let bytes: Vec<u8> = obj.call_method1("to_bytes", (byte_len, "little"))?.extract()?;
    Ok(Bitset::from_le_bytes(&bytes))
}

pub fn bitset_to_pyint(py: Python<'_>, bs: &Bitset) -> PyResult<PyObject> {
    let bytes = bs.to_le_bytes();
    let builtins = py.import("builtins")?;
    let int_type = builtins.getattr("int")?;
    Ok(int_type.call_method1("from_bytes", (bytes.as_slice(), "little"))?.into())
}
