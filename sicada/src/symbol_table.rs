use std::collections::BTreeMap;
use std::fmt::Write as FmtWrite;
use std::hash::{Hash, Hasher};
use std::io::{BufRead, Read, Write};

use md5::{Digest, Md5};
use rustc_hash::FxHasher;

use crate::AtomicRc;
use crate::error::OpenFstError;
use crate::utils::io::{read_scalar, read_string, write_scalar, write_string};

pub const K_NO_SYMBOL: i64 = -1;
pub const K_SYMBOL_TABLE_MAGIC_NUMBER: i32 = 2125658996;

/// Computes the checksum of symbol table entries, used by `compat_symbols` to
/// decide whether two tables agree.
///
/// SICADA-DIVERGE: upstream's `CheckSummer` (`compat/checksummer.h`, pulled in by
/// `compat-util.h`) XORs each byte into a 32-byte buffer at `count % 32`. Any two
/// byte streams that agree per residue class mod 32 collide, and swapping two
/// bytes 32 positions apart is enough, so `CompatSymbols` accepts symbol tables
/// that do not match. We use MD5 instead. Checksums are never serialized,
/// only compared between two in-memory tables, so this cannot affect
/// compatibility with FSTs written by OpenFst.
struct CheckSummer(Md5);

impl CheckSummer {
    fn new() -> Self {
        Self(Md5::new())
    }

    fn update(&mut self, data: impl AsRef<[u8]>) {
        self.0.update(data);
    }

    /// Finalizes the MD5 digest and formats it as a 32-character hexadecimal string.
    fn digest(self) -> String {
        let digest = self.0.finalize();
        let mut hex_string = String::with_capacity(32);
        for byte in digest {
            write!(&mut hex_string, "{:02x}", byte).expect("String formatting should never fail");
        }
        hex_string
    }
}

const K_EMPTY_BUCKET: isize = -1;
const K_MAX_OCCUPANCY_RATIO: f32 = 0.75;

/// An open-addressed, linear probing hash map optimized specifically for symbol strings.
///
/// This structure completely avoids duplicating string allocations by keeping
/// a single contiguous `Vec<String>` and using a separate `Vec<isize>` for hash buckets.
/// It provides near O(1) lookups and significantly reduces memory overhead.
#[derive(Debug, Clone)]
struct DenseSymbolMap {
    symbols: Vec<String>,
    buckets: Vec<isize>,
    hash_mask: usize,
}

impl DenseSymbolMap {
    fn new() -> Self {
        let initial_buckets = 16;
        Self {
            symbols: Vec::new(),
            buckets: vec![K_EMPTY_BUCKET; initial_buckets],
            hash_mask: initial_buckets - 1,
        }
    }

    /// Computes the hash of a symbol string using the high-performance FxHasher.
    fn get_hash(key: &str) -> u64 {
        let mut hasher = FxHasher::default();
        key.hash(&mut hasher);
        hasher.finish()
    }

    /// Inserts a string into the map if it does not already exist.
    /// Returns a tuple containing the internal index of the symbol and a boolean
    /// indicating whether a new entry was inserted.
    fn insert_or_find(&mut self, key: &str) -> (usize, bool) {
        if self.symbols.len() as f32 >= (self.buckets.len() as f32 * K_MAX_OCCUPANCY_RATIO) {
            self.rehash(self.buckets.len() * 2);
        }

        let mut idx = (Self::get_hash(key) as usize) & self.hash_mask;
        loop {
            let stored_value = self.buckets[idx];
            if stored_value == K_EMPTY_BUCKET {
                break;
            }
            if self.symbols[stored_value as usize] == key {
                return (stored_value as usize, false);
            }
            idx = (idx + 1) & self.hash_mask;
        }

        let next = self.symbols.len();
        self.buckets[idx] = next as isize;
        self.symbols.push(key.to_string());
        (next, true)
    }

    /// Returns the internal index of the given symbol, or `K_EMPTY_BUCKET` if not found.
    fn find(&self, key: &str) -> isize {
        let mut idx = (Self::get_hash(key) as usize) & self.hash_mask;
        loop {
            let stored_value = self.buckets[idx];
            if stored_value == K_EMPTY_BUCKET {
                return K_EMPTY_BUCKET;
            }
            if self.symbols[stored_value as usize] == key {
                return stored_value;
            }
            idx = (idx + 1) & self.hash_mask;
        }
    }

    fn size(&self) -> usize {
        self.symbols.len()
    }

    fn get_symbol(&self, idx: usize) -> &str {
        &self.symbols[idx]
    }

    /// Resizes the hash bucket array and recalculates positions for all elements.
    fn rehash(&mut self, num_buckets: usize) {
        assert!(num_buckets.is_power_of_two());
        self.buckets.clear();
        self.buckets.resize(num_buckets, K_EMPTY_BUCKET);
        self.hash_mask = num_buckets - 1;

        for (i, symbol) in self.symbols.iter().enumerate() {
            let mut idx = (Self::get_hash(symbol) as usize) & self.hash_mask;
            while self.buckets[idx] != K_EMPTY_BUCKET {
                idx = (idx + 1) & self.hash_mask;
            }
            self.buckets[idx] = i as isize;
        }
    }

    /// Removes a symbol by its internal index, forcing a rehash to repair the probe sequence.
    fn remove_symbol(&mut self, idx: usize) {
        self.symbols.remove(idx);
        self.rehash(self.buckets.len());
    }

    /// Minimizes the memory footprint of the internal structures based on current occupancy.
    fn shrink_to_fit(&mut self) {
        self.symbols.shrink_to_fit();
        let required_capacity = (self.symbols.len() as f32 / K_MAX_OCCUPANCY_RATIO) as usize;
        let mut new_buckets = 16;
        while new_buckets < required_capacity {
            new_buckets *= 2;
        }
        if new_buckets < self.buckets.len() {
            self.rehash(new_buckets);
        }
        self.buckets.shrink_to_fit();
    }
}

/// The core implementation of the SymbolTable mapping.
///
/// This structure faithfully implements OpenFst's hybrid dense/sparse indexing strategy:
/// Sequential keys `[0, dense_key_limit)` are mapped implicitly via contiguous arrays (`O(1)`).
/// Non-sequential or negative keys are offloaded to a `BTreeMap` (`O(log N)`).
#[derive(Debug, Clone)]
struct SymbolTableImpl {
    name: String,
    available_key: i64,
    dense_key_limit: i64,

    symbols: DenseSymbolMap,
    idx_key: Vec<i64>,
    key_map: BTreeMap<i64, i64>,

    check_sum_finalized: bool,
    check_sum_string: String,
    labeled_check_sum_string: String,
}

impl SymbolTableImpl {
    fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            available_key: 0,
            dense_key_limit: 0,
            symbols: DenseSymbolMap::new(),
            idx_key: Vec::new(),
            key_map: BTreeMap::new(),
            check_sum_finalized: false,
            check_sum_string: String::new(),
            labeled_check_sum_string: String::new(),
        }
    }

    /// Associates a symbol string with a given key.
    fn add_symbol(&mut self, symbol: &str, key: i64) -> i64 {
        if key == K_NO_SYMBOL {
            return key;
        }

        let (insert_idx, inserted) = self.symbols.insert_or_find(symbol);
        if !inserted {
            let key_already = self.get_nth_key(insert_idx as isize);
            return if key_already == key { key } else { key_already };
        }

        if key + 1 == self.symbols.size() as i64 && key == self.dense_key_limit {
            self.dense_key_limit += 1;
        } else {
            self.idx_key.push(key);
            self.key_map.insert(key, self.symbols.size() as i64 - 1);
        }

        if key >= self.available_key {
            self.available_key = key + 1;
        }

        self.check_sum_finalized = false;
        key
    }

    /// Associates a symbol string with the next sequential available key.
    fn add_symbol_auto(&mut self, symbol: &str) -> i64 {
        self.add_symbol(symbol, self.available_key)
    }

    /// Erases a key and its associated symbol from the table.
    fn remove_symbol(&mut self, key: i64) {
        let mut idx = key;

        if key < 0 || key >= self.dense_key_limit {
            if let Some(&mapped_idx) = self.key_map.get(&key) {
                idx = mapped_idx;
                self.key_map.remove(&key);
            } else {
                return;
            }
        }

        if idx < 0 || idx >= self.symbols.size() as i64 {
            return;
        }

        self.symbols.remove_symbol(idx as usize);

        for mapped_idx in self.key_map.values_mut() {
            if *mapped_idx > idx {
                *mapped_idx -= 1;
            }
        }

        if key >= 0 && key < self.dense_key_limit {
            let new_dense_key_limit = key;

            for i in (key + 1)..self.dense_key_limit {
                self.key_map.insert(i, i - 1);
            }

            let old_idx_key_len = self.idx_key.len();
            self.idx_key
                .resize(self.symbols.size() - new_dense_key_limit as usize, 0);

            // Backwards iteration to prevent overwriting uncopied data when shifting.
            for i in (self.dense_key_limit as usize..=self.symbols.size()).rev() {
                let dest_idx = i - new_dense_key_limit as usize - 1;
                let src_idx = i - self.dense_key_limit as usize;

                if src_idx < old_idx_key_len {
                    self.idx_key[dest_idx] = self.idx_key[src_idx];
                } else {
                    self.idx_key[dest_idx] = 0;
                }
            }

            for i in new_dense_key_limit..(self.dense_key_limit - 1) {
                self.idx_key[(i - new_dense_key_limit) as usize] = i + 1;
            }

            self.dense_key_limit = new_dense_key_limit;
        } else {
            let start_idx = (idx - self.dense_key_limit) as usize;
            for i in start_idx..(self.idx_key.len() - 1) {
                self.idx_key[i] = self.idx_key[i + 1];
            }
            self.idx_key.pop();
        }

        if key == self.available_key - 1 {
            self.available_key = key;
        }

        self.check_sum_finalized = false;
    }

    /// Resolves the symbol string associated with a given key.
    fn find_symbol(&self, key: i64) -> Option<&str> {
        let idx = if key < 0 || key >= self.dense_key_limit {
            *self.key_map.get(&key)?
        } else {
            key
        };

        if idx >= 0 && (idx as usize) < self.symbols.size() {
            Some(self.symbols.get_symbol(idx as usize))
        } else {
            None
        }
    }

    /// Resolves the integer key associated with a given symbol string.
    fn find_key(&self, symbol: &str) -> i64 {
        let idx = self.symbols.find(symbol);
        if idx == K_EMPTY_BUCKET {
            return K_NO_SYMBOL;
        }
        if idx < self.dense_key_limit as isize {
            return idx as i64;
        }
        self.idx_key[(idx - self.dense_key_limit as isize) as usize]
    }

    /// Retrieves the exact key at a given internal memory index.
    fn get_nth_key(&self, pos: isize) -> i64 {
        if pos < 0 || pos as usize >= self.symbols.size() {
            K_NO_SYMBOL
        } else if pos < self.dense_key_limit as isize {
            pos as i64
        } else {
            self.find_key(self.symbols.get_symbol(pos as usize))
        }
    }

    /// Computes both legacy and labeled MD5 checksums of the table structure.
    fn maybe_recompute_check_sum(&mut self) {
        if self.check_sum_finalized {
            return;
        }

        let mut check_sum = CheckSummer::new();
        for i in 0..self.symbols.size() {
            check_sum.update(self.symbols.get_symbol(i).as_bytes());
            check_sum.update(b"\0");
        }
        self.check_sum_string = check_sum.digest();

        let mut labeled_check_sum = CheckSummer::new();
        for i in 0..self.dense_key_limit {
            labeled_check_sum.update(format!("{}\t{}", self.symbols.get_symbol(i as usize), i));
        }
        for (&key, &idx) in &self.key_map {
            // SICADA-BUGFIX: upstream skips every key below `dense_key_limit_`,
            // which is meant to avoid repeating what the dense loop above already
            // covered, but negative keys are below it too, so they are dropped
            // from the checksum entirely. Its own comment calls this a bug kept
            // because "too many tests rely on" it. Two tables differing only in
            // their negatively labelled symbols then get the same labelled
            // checksum and `compat_symbols` accepts them as equivalent. The
            // checksum is never serialized, so correcting it cannot affect
            // compatibility with files OpenFst wrote.
            if (0..self.dense_key_limit).contains(&key) {
                continue;
            }
            labeled_check_sum.update(format!(
                "{}\t{}",
                self.symbols.get_symbol(idx as usize),
                key
            ));
        }
        self.labeled_check_sum_string = labeled_check_sum.digest();
        self.check_sum_finalized = true;
    }

    /// Serializes the table to an OpenFst-compatible binary format.
    ///
    /// A magic number, the table name, the next available key and the symbol
    /// count, then that many `(symbol, key)` pairs. Strings carry a 32-bit length
    /// prefix and keys are 64-bit; see `utils::io`.
    fn write<W: Write>(&self, w: &mut W) -> Result<(), OpenFstError> {
        write_scalar(w, K_SYMBOL_TABLE_MAGIC_NUMBER)?;
        write_string(w, &self.name)?;
        write_scalar(w, self.available_key)?;
        write_scalar(w, self.symbols.size() as i64)?;

        for i in 0..self.dense_key_limit {
            write_string(w, self.symbols.get_symbol(i as usize))?;
            write_scalar(w, i)?;
        }

        // SICADA-DIVERGE: upstream walks a hash map here, so the order of the
        // sparse entries, and therefore the bytes of the file, depends on the
        // hash implementation and on insertion history. Walking an ordered map
        // makes the output reproducible; the reader does not care about order.
        for (&key, &idx) in &self.key_map {
            write_string(w, self.symbols.get_symbol(idx as usize))?;
            write_scalar(w, key)?;
        }

        w.flush()?;
        Ok(())
    }

    /// Deserializes the table from an OpenFst-compatible binary format.
    fn read<R: Read>(r: &mut R) -> Result<Self, OpenFstError> {
        let magic: i32 = read_scalar(r)?;
        if magic != K_SYMBOL_TABLE_MAGIC_NUMBER {
            return Err(OpenFstError::SymbolTable(format!(
                "Invalid symbol table magic number: expected {}, found {}",
                K_SYMBOL_TABLE_MAGIC_NUMBER, magic
            )));
        }

        let name = read_string(r)?;
        let mut table = Self::new(name);
        table.available_key = read_scalar(r)?;
        let size: i64 = read_scalar(r)?;
        if size < 0 {
            return Err(OpenFstError::SymbolTable(format!(
                "Invalid symbol table size: {size}"
            )));
        }

        table.check_sum_finalized = false;
        for _ in 0..size {
            let symbol = read_string(r)?;
            let key: i64 = read_scalar(r)?;
            table.add_symbol(&symbol, key);
        }

        table.symbols.shrink_to_fit();
        Ok(table)
    }
}

/// A reference-counted, copy-on-write mapping of strings to integer labels.
///
/// `SymbolTable` defines the alphabet of the input and output labels for arcs
/// in a Finite State Transducer. It supports both sequential implicit assignment
/// and manual sparse assignments.
#[derive(Debug, Clone)]
pub struct SymbolTable {
    inner: AtomicRc<SymbolTableImpl>,
}

impl SymbolTable {
    /// Creates a new, empty SymbolTable with a given identifying name.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            inner: AtomicRc::new(SymbolTableImpl::new(name)),
        }
    }

    /// Provides mutable access to the underlying table.
    /// Clones the implementation if another FST is currently referencing it.
    fn make_mut(&mut self) -> &mut SymbolTableImpl {
        std::sync::Arc::make_mut(&mut self.inner)
    }

    /// Reads an OpenFst-compatible binary symbol table dump.
    pub fn read<R: Read>(reader: &mut R) -> Result<Self, OpenFstError> {
        let inner = SymbolTableImpl::read(reader)?;
        Ok(Self {
            inner: AtomicRc::new(inner),
        })
    }

    /// Writes the symbol table as an OpenFst-compatible binary format.
    pub fn write<W: Write>(&self, writer: &mut W) -> Result<(), OpenFstError> {
        self.inner.write(writer)
    }

    /// Parses a text representation of the symbol table.
    /// The separator can be configured via `sep`. An empty string defaults to "\t ".
    pub fn read_text<R: BufRead>(
        reader: &mut R,
        name: impl Into<String>,
        sep: &str,
    ) -> Result<Self, OpenFstError> {
        let mut table = SymbolTableImpl::new(name);
        let default_sep = "\t ";
        let separator = if sep.is_empty() { default_sep } else { sep };

        let mut line = String::new();
        let mut nline = 0;

        while let Ok(bytes) = reader.read_line(&mut line) {
            if bytes == 0 {
                break;
            }
            nline += 1;
            let trimmed = line.trim_end_matches(&['\n', '\r'][..]);
            if trimmed.is_empty() {
                line.clear();
                continue;
            }

            let parts: Vec<&str> = trimmed
                .split(|c| separator.contains(c))
                .filter(|s| !s.is_empty())
                .collect();

            if parts.len() != 2 {
                return Err(OpenFstError::SymbolTable(format!(
                    "ReadText: Bad number of columns ({}), line = {}",
                    parts.len(),
                    nline
                )));
            }

            let symbol = parts[0];
            let key = parts[1].parse::<i64>().map_err(|_| {
                OpenFstError::SymbolTable(format!(
                    "ReadText: Invalid integer label ({}), line = {}",
                    parts[1], nline
                ))
            })?;

            table.add_symbol(symbol, key);
            line.clear();
        }

        table.symbols.shrink_to_fit();
        Ok(Self {
            inner: AtomicRc::new(table),
        })
    }

    /// Writes a text representation of the symbol table line by line.
    /// The separator can be configured via `sep`. An empty string defaults to "\t".
    pub fn write_text<W: Write>(&self, writer: &mut W, sep: &str) -> Result<(), OpenFstError> {
        let default_sep = "\t";
        let separator = if sep.is_empty() { default_sep } else { sep };

        for item in self.iter() {
            writeln!(writer, "{}{}{}", item.symbol, separator, item.label)?;
        }
        writer.flush()?;
        Ok(())
    }

    /// Inserts a symbol strictly mapped to the provided integer key.
    pub fn add_symbol(&mut self, symbol: &str, key: i64) -> i64 {
        self.make_mut().add_symbol(symbol, key)
    }

    /// Inserts a symbol and automatically provisions the next contiguous available key.
    pub fn add_symbol_auto(&mut self, symbol: &str) -> i64 {
        self.make_mut().add_symbol_auto(symbol)
    }

    /// Deletes a mapping based on its integer key.
    pub fn remove_symbol(&mut self, key: i64) {
        self.make_mut().remove_symbol(key)
    }

    /// Looks up the string representation bound to a given integer key.
    pub fn find_symbol(&self, key: i64) -> Option<&str> {
        self.inner.find_symbol(key)
    }

    /// Looks up the integer key bound to a given string representation.
    pub fn find_key(&self, symbol: &str) -> i64 {
        self.inner.find_key(symbol)
    }

    /// Checks if a key is registered in the table.
    pub fn member_key(&self, key: i64) -> bool {
        self.find_symbol(key).is_some()
    }

    /// Checks if a symbol string is registered in the table.
    pub fn member_symbol(&self, symbol: &str) -> bool {
        self.find_key(symbol) != K_NO_SYMBOL
    }

    /// Returns the smallest positive contiguous key not yet assigned.
    pub fn available_key(&self) -> i64 {
        self.inner.available_key
    }

    /// Returns the logical name assigned to the table.
    pub fn name(&self) -> &str {
        &self.inner.name
    }

    /// Reassigns the logical name of the table.
    pub fn set_name(&mut self, name: impl Into<String>) {
        self.make_mut().name = name.into();
    }

    /// Returns the total count of bound symbols.
    pub fn num_symbols(&self) -> usize {
        self.inner.symbols.size()
    }

    /// Internally resolves the positional key at `pos`.
    fn get_nth_key(&self, pos: isize) -> i64 {
        self.inner.get_nth_key(pos)
    }

    /// Returns an MD5 checksum over the symbols (legacy).
    pub fn check_sum(&mut self) -> &str {
        self.make_mut().maybe_recompute_check_sum();
        &self.inner.check_sum_string
    }

    /// Returns an MD5 checksum strongly coupling symbols and keys.
    pub fn labeled_check_sum(&mut self) -> &str {
        self.make_mut().maybe_recompute_check_sum();
        &self.inner.labeled_check_sum_string
    }

    /// Appends all symbols from an external table, assigning sequential auto-keys if needed.
    pub fn add_table(&mut self, table: &SymbolTable) {
        let mut_impl = self.make_mut();
        for item in table.iter() {
            mut_impl.add_symbol_auto(&item.symbol);
        }
    }

    /// Provides a standard Rust iterator over the (label, symbol) entries.
    pub fn iter(&self) -> SymbolTableIterator<'_> {
        SymbolTableIterator {
            table: self,
            pos: 0,
            nsymbols: self.num_symbols(),
        }
    }
}

/// Represents an individual resolved record generated during table iteration.
pub struct SymbolTableItem {
    pub label: i64,
    pub symbol: String,
}

/// An iterator yielding sequential and sparse `SymbolTableItem` records.
pub struct SymbolTableIterator<'a> {
    table: &'a SymbolTable,
    pos: usize,
    nsymbols: usize,
}

impl<'a> Iterator for SymbolTableIterator<'a> {
    type Item = SymbolTableItem;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos < self.nsymbols {
            let key = self.table.get_nth_key(self.pos as isize);
            let symbol = self.table.find_symbol(key).unwrap().to_string();
            self.pos += 1;
            Some(SymbolTableItem { label: key, symbol })
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let rem = self.nsymbols - self.pos;
        (rem, Some(rem))
    }
}

/// Asserts the structural/mathematical compatibility between two symbol tables.
pub fn compat_symbols(syms1: Option<&mut SymbolTable>, syms2: Option<&mut SymbolTable>) -> bool {
    if let (Some(s1), Some(s2)) = (syms1, syms2)
        && s1.labeled_check_sum() != s2.labeled_check_sum()
    {
        return false;
    }
    true
}

/// Asserts the structural/mathematical compatibility between two symbol tables.
pub fn compat_symbols_rc(
    mut syms1: Option<AtomicRc<SymbolTable>>,
    mut syms2: Option<AtomicRc<SymbolTable>>,
) -> bool {
    compat_symbols(
        syms1.as_mut().map(AtomicRc::make_mut),
        syms2.as_mut().map(AtomicRc::make_mut),
    )
}

pub fn compat_symbols_with_warn(
    syms1: Option<&mut SymbolTable>,
    syms2: Option<&mut SymbolTable>,
    warning: &mut dyn Write,
) -> Result<bool, std::io::Error> {
    if let (Some(s1), Some(s2)) = (syms1, syms2)
        && s1.labeled_check_sum() != s2.labeled_check_sum()
    {
        writeln!(
            warning,
            "WARNING: CompatSymbols: Symbol table checksums do not match. Table sizes are {} and {}",
            s1.num_symbols(),
            s2.num_symbols()
        )?;
        return Ok(false);
    }
    Ok(true)
}
