//!
//! Persisting a term sum to disk and reading it back (term file format v2).
//!
//! All integers are little-endian. The file is gzip-compressed.
//!
//! | Field | Type |
//! |---|---|
//! | magic | 8 bytes `PQTERMS\0` |
//! | version | u32 = 2 |
//! | basis kind | u32 (`BasisKind`) |
//! | IR hash | 16 ASCII bytes, zeros if none |
//! | bits per unit | u32 |
//! | n_units | u64 |
//! | words per key | u32 |
//! | coefficient type | u32: 0 = f64, 1 = complex128 |
//! | n_terms | u64 |
//! | terms | key as `words per key` u64 words, then the coefficient |
//!
//! A key uses the canonical `BasisString` layout. Unit `u` owns the bits
//! `u * bits_per_unit ..  (u + 1) * bits_per_unit`. A coefficient is one f64,
//! or two f64 (real, imaginary) for complex128.
//!
//! A file without the magic is a legacy v0.1.8 file. It holds
//! `n_terms`, `key_stride` and `system_size` (3 x u64), then raw key bytes
//! and one f64 per term. The reader converts it to the canonical layout.
//! The reader needs the kind of the basis for this (Pauli or Majorana).
//!
//! Writers always write v2.
//!
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};

use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;
use num_complex::Complex64;
use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use pyo3_stub_gen::{PyStubType, TypeInfo};

use crate::basis::BasisKind;

/// The magic number at the start of a v2 file.
pub const MAGIC: [u8; 8] = *b"PQTERMS\0";
/// The format version that writers produce.
pub const VERSION: u32 = 2;
/// Coefficient type code: one f64 per term.
pub const COEFF_F64: u32 = 0;
/// Coefficient type code: two f64 (real, imaginary) per term.
pub const COEFF_C128: u32 = 1;

/// IR hashes of the built-in Pauli and Majorana programs.
///
/// A traced module that runs one of these programs is the same basis as the
/// built-in one. Its files carry the built-in kind and a zero hash.
/// A test (`tests/io/test_term_file_ids.py`) keeps these values in step
/// with the programs.
pub const INTRINSIC_HASHES: [(&str, BasisKind); 2] = [
    ("8c78739211713c67", BasisKind::Pauli),
    ("47024e944beb662a", BasisKind::Majorana),
];

/// Which basis a term file belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BasisId {
    /// `BasisKind` as u32.
    pub kind: u32,
    /// 16 ASCII hex bytes of the IR hash, or zeros.
    pub ir_hash: [u8; 16],
    /// Bits per unit of the canonical layout.
    pub bits_per_unit: u32,
}

impl BasisId {
    /// An id with a zero hash.
    pub fn new(kind: BasisKind, bits_per_unit: usize) -> Self {
        BasisId { kind: kind.as_u32(), ir_hash: [0; 16], bits_per_unit: bits_per_unit as u32 }
    }

    /// The same id with a hash. `hash` must hold 16 ASCII bytes; other input gives a zero hash.
    pub fn with_hash(mut self, hash: &str) -> Self {
        self.ir_hash = if hash.len() == 16 && hash.is_ascii() {
            let mut h = [0u8; 16];
            h.copy_from_slice(hash.as_bytes());
            h
        } else {
            [0; 16]
        };
        self
    }

    /// The canonical form. A traced copy of a built-in program becomes the built-in id.
    pub fn normalized(mut self) -> Self {
        if self.kind == BasisKind::Custom.as_u32() {
            for (hash, kind) in INTRINSIC_HASHES {
                if self.ir_hash == hash.as_bytes() {
                    self.kind = kind.as_u32();
                    self.ir_hash = [0; 16];
                }
            }
        }
        self
    }

    fn kind_name(&self) -> String {
        match self.kind {
            0 => "Pauli".to_string(),
            1 => "Majorana".to_string(),
            2 => "custom".to_string(),
            k => format!("unknown kind {k}"),
        }
    }

    fn hash_text(&self) -> String {
        if self.ir_hash == [0; 16] {
            "none".to_string()
        } else {
            String::from_utf8_lossy(&self.ir_hash).into_owned()
        }
    }

    /// A short text for error messages.
    pub fn describe(&self) -> String {
        format!(
            "{} basis (IR hash {}, {} bits per unit)",
            self.kind_name(),
            self.hash_text(),
            self.bits_per_unit
        )
    }
}

/// A coefficient as the file holds it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FileCoeff {
    Real(f64),
    Complex(Complex64),
}

impl FileCoeff {
    /// The value as a complex number.
    pub fn to_complex(self) -> Complex64 {
        match self {
            FileCoeff::Real(r) => Complex64::new(r, 0.0),
            FileCoeff::Complex(z) => z,
        }
    }
}

impl<'py> IntoPyObject<'py> for FileCoeff {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        match self {
            FileCoeff::Real(r) => Ok(r.into_pyobject(py)?.into_any()),
            FileCoeff::Complex(z) => Ok(z.into_pyobject(py)?.into_any()),
        }
    }
}

impl PyStubType for FileCoeff {
    fn type_output() -> TypeInfo {
        TypeInfo::builtin("float | complex")
    }
    fn type_input() -> TypeInfo {
        TypeInfo::builtin("float | complex")
    }
}

/// The header of a term file, after any legacy conversion.
#[derive(Clone, Debug)]
pub struct TermFileHeader {
    pub id: BasisId,
    pub n_units: u64,
    pub words_per_key: u32,
    pub complex: bool,
    pub n_terms: u64,
    /// True if the file was a legacy v0.1.8 file.
    pub legacy: bool,
}

/// Number of words in a key of `n_units` units (canonical layout).
pub fn words_for(n_units: u64, bits_per_unit: u32) -> u64 {
    (n_units * u64::from(bits_per_unit)).div_ceil(64).max(1)
}

fn io_err(what: &str, e: std::io::Error) -> PyErr {
    if e.kind() == std::io::ErrorKind::UnexpectedEof {
        PyIOError::new_err(format!("term file is truncated or corrupt ({what})"))
    } else {
        PyIOError::new_err(format!("term file read error ({what}): {e}"))
    }
}

fn bad(msg: String) -> PyErr {
    PyValueError::new_err(msg)
}

// ------------------------------------------------------------------ writer

/// Writes a v2 term file. Call `write_term` exactly `n_terms` times, then `finish`.
pub struct TermFileWriter {
    enc: GzEncoder<BufWriter<File>>,
    words_per_key: usize,
    complex: bool,
    left: u64,
}

impl TermFileWriter {
    pub fn create(path: &str, header: &TermFileHeader) -> PyResult<Self> {
        let file = std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(path)
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        let mut enc = GzEncoder::new(BufWriter::new(file), Compression::default());
        let mut head = Vec::with_capacity(76);
        head.extend_from_slice(&MAGIC);
        head.extend_from_slice(&VERSION.to_le_bytes());
        head.extend_from_slice(&header.id.kind.to_le_bytes());
        head.extend_from_slice(&header.id.ir_hash);
        head.extend_from_slice(&header.id.bits_per_unit.to_le_bytes());
        head.extend_from_slice(&header.n_units.to_le_bytes());
        head.extend_from_slice(&header.words_per_key.to_le_bytes());
        let coeff_type = if header.complex { COEFF_C128 } else { COEFF_F64 };
        head.extend_from_slice(&coeff_type.to_le_bytes());
        head.extend_from_slice(&header.n_terms.to_le_bytes());
        enc.write_all(&head).map_err(|e| PyIOError::new_err(e.to_string()))?;
        Ok(TermFileWriter {
            enc,
            words_per_key: header.words_per_key as usize,
            complex: header.complex,
            left: header.n_terms,
        })
    }

    pub fn write_term(&mut self, words: &[u64], coeff: FileCoeff) -> PyResult<()> {
        if words.len() != self.words_per_key || self.left == 0 {
            return Err(PyValueError::new_err("term file writer got a key of the wrong size"));
        }
        self.left -= 1;
        let mut buf = Vec::with_capacity(8 * (words.len() + 2));
        for w in words {
            buf.extend_from_slice(&w.to_le_bytes());
        }
        let z = coeff.to_complex();
        buf.extend_from_slice(&z.re.to_le_bytes());
        if self.complex {
            buf.extend_from_slice(&z.im.to_le_bytes());
        }
        self.enc.write_all(&buf).map_err(|e| PyIOError::new_err(e.to_string()))
    }

    pub fn finish(self) -> PyResult<()> {
        if self.left != 0 {
            return Err(PyValueError::new_err("term file writer got fewer terms than declared"));
        }
        self.enc.finish().map_err(|e| PyIOError::new_err(e.to_string()))?;
        Ok(())
    }
}

// ------------------------------------------------------------------ reader

#[derive(Clone, Copy, PartialEq)]
enum Layout {
    V2,
    LegacyPauli { key_stride: usize, n: usize },
    LegacyMajorana { key_stride: usize, modes: usize },
}

/// Reads a term file, v2 or legacy. Keys come out as canonical words.
pub struct TermFileReader {
    reader: BufReader<GzDecoder<File>>,
    pub header: TermFileHeader,
    layout: Layout,
    left: u64,
    buf: Vec<u8>,
}

fn read_u64(r: &mut impl Read, what: &str) -> PyResult<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b).map_err(|e| io_err(what, e))?;
    Ok(u64::from_le_bytes(b))
}

fn read_u32(r: &mut impl Read, what: &str) -> PyResult<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b).map_err(|e| io_err(what, e))?;
    Ok(u32::from_le_bytes(b))
}

impl TermFileReader {
    /// Opens `path` for a basis with id `expect`.
    ///
    /// A v2 file must match `expect` in kind, IR hash and bits per unit
    /// (both ids are normalized first). A legacy file needs `expect` to be
    /// Pauli or Majorana.
    pub fn open(path: &str, expect: &BasisId) -> PyResult<Self> {
        let file = File::open(path).map_err(|e| PyIOError::new_err(e.to_string()))?;
        let mut reader = BufReader::new(GzDecoder::new(file));
        let first = read_u64(&mut reader, "header")?;
        let expect = expect.normalized();
        if first.to_le_bytes() == MAGIC {
            Self::open_v2(reader, &expect)
        } else {
            Self::open_legacy(reader, first, &expect)
        }
    }

    fn open_v2(mut reader: BufReader<GzDecoder<File>>, expect: &BasisId) -> PyResult<Self> {
        let version = read_u32(&mut reader, "version")?;
        if version != VERSION {
            return Err(bad(format!(
                "term file has format version {version}. This propaq reads version {VERSION} \
                 and legacy v0.1.8 files."
            )));
        }
        let kind = read_u32(&mut reader, "basis kind")?;
        let mut ir_hash = [0u8; 16];
        reader.read_exact(&mut ir_hash).map_err(|e| io_err("IR hash", e))?;
        let bits = read_u32(&mut reader, "bits per unit")?;
        let n_units = read_u64(&mut reader, "n_units")?;
        let words_per_key = read_u32(&mut reader, "words per key")?;
        let coeff_type = read_u32(&mut reader, "coefficient type")?;
        let n_terms = read_u64(&mut reader, "n_terms")?;
        let id = BasisId { kind, ir_hash, bits_per_unit: bits }.normalized();
        if id != *expect {
            return Err(bad(format!(
                "term file holds terms of a {}. This term sum is a {}.",
                id.describe(),
                expect.describe()
            )));
        }
        if coeff_type != COEFF_F64 && coeff_type != COEFF_C128 {
            return Err(bad(format!("term file has an unknown coefficient type {coeff_type}")));
        }
        if bits == 0 || u64::from(words_per_key) != words_for(n_units, bits) {
            return Err(bad(format!(
                "term file header is inconsistent: {n_units} units of {bits} bits \
                 do not fit {words_per_key} words per key"
            )));
        }
        let complex = coeff_type == COEFF_C128;
        let header = TermFileHeader { id, n_units, words_per_key, complex, n_terms, legacy: false };
        Ok(TermFileReader {
            reader,
            header,
            layout: Layout::V2,
            left: n_terms,
            buf: vec![0u8; 8 * words_per_key as usize],
        })
    }

    fn open_legacy(
        mut reader: BufReader<GzDecoder<File>>,
        n_terms: u64,
        expect: &BasisId,
    ) -> PyResult<Self> {
        let key_stride = read_u64(&mut reader, "legacy header")? as usize;
        let system_size = read_u64(&mut reader, "legacy header")? as usize;
        let kind = expect.kind;
        let empty = n_terms == 0;
        let (layout, n_units) = if kind == BasisKind::Pauli.as_u32() {
            if !empty && key_stride != 2 * system_size.div_ceil(8) {
                return Err(bad(format!(
                    "legacy term file does not hold Pauli strings: key size {key_stride} bytes \
                     for {system_size} qubits"
                )));
            }
            (Layout::LegacyPauli { key_stride, n: system_size }, system_size as u64)
        } else if kind == BasisKind::Majorana.as_u32() {
            if !empty && key_stride != system_size.div_ceil(8) {
                return Err(bad(format!(
                    "legacy term file does not hold Majorana monomials: key size {key_stride} \
                     bytes for {system_size} modes"
                )));
            }
            if !system_size.is_multiple_of(2) {
                return Err(bad(format!(
                    "legacy Majorana term file has an odd mode count ({system_size})"
                )));
            }
            (Layout::LegacyMajorana { key_stride, modes: system_size }, (system_size / 2) as u64)
        } else {
            return Err(bad(format!(
                "term file is a legacy v0.1.8 file. It has no basis id, so a {} cannot read it. \
                 Only Pauli and Majorana term sums read legacy files.",
                expect.describe()
            )));
        };
        let words_per_key = words_for(n_units, expect.bits_per_unit) as u32;
        let header = TermFileHeader {
            id: *expect,
            n_units,
            words_per_key,
            complex: false,
            n_terms,
            legacy: true,
        };
        Ok(TermFileReader {
            reader,
            header,
            layout,
            left: n_terms,
            buf: vec![0u8; key_stride],
        })
    }

    /// The next term as canonical words and a coefficient, or `None` at the end.
    pub fn next_term(&mut self) -> PyResult<Option<(Vec<u64>, FileCoeff)>> {
        if self.left == 0 {
            return Ok(None);
        }
        self.left -= 1;
        self.reader.read_exact(&mut self.buf).map_err(|e| io_err("term key", e))?;
        let words = match self.layout {
            Layout::V2 => {
                let words: Vec<u64> =
                    self.buf.as_chunks::<8>().0.iter().map(|c| u64::from_le_bytes(*c)).collect();
                let used = self.header.n_units * u64::from(self.header.id.bits_per_unit);
                let tail = used % 64;
                if tail != 0 && words.last().is_some_and(|w| w >> tail != 0) {
                    return Err(bad("term file is corrupt: a key has bits past its last unit".into()));
                }
                words
            }
            Layout::LegacyPauli { key_stride, n } => {
                let half = key_stride / 2;
                let mut words = vec![0u64; self.header.words_per_key as usize];
                for q in 0..n {
                    let x = self.buf[q / 8] >> (q % 8) & 1 != 0;
                    let z = self.buf[half + q / 8] >> (q % 8) & 1 != 0;
                    words[(2 * q) / 64] |= u64::from(x) << ((2 * q) % 64);
                    words[(2 * q + 1) / 64] |= u64::from(z) << ((2 * q + 1) % 64);
                }
                words
            }
            Layout::LegacyMajorana { modes, .. } => {
                let mut words = vec![0u64; self.header.words_per_key as usize];
                for i in 0..modes {
                    words[i / 64] |= u64::from(self.buf[i / 8] >> (i % 8) & 1) << (i % 64);
                }
                words
            }
        };
        let re = read_f64(&mut self.reader)?;
        let coeff = if self.header.complex {
            FileCoeff::Complex(Complex64::new(re, read_f64(&mut self.reader)?))
        } else {
            FileCoeff::Real(re)
        };
        Ok(Some((words, coeff)))
    }
}

fn read_f64(r: &mut impl Read) -> PyResult<f64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b).map_err(|e| io_err("coefficient", e))?;
    Ok(f64::from_le_bytes(b))
}
