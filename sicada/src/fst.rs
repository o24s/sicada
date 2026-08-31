//! The FST interface, and the options and shared header handling around it.
//!
//! Port of OpenFst's `fst.h` and `fst.cc`.
//!
//! # Where the command-line flags went
//!
//! `fst.cc` defines seven ABSL flags. Six of them exist only to supply a default
//! to a field that the caller can already set: `fst_align` to
//! [`FstWriteOptions::align`], `fst_read_mode` to [`FstReadOptions::mode`],
//! `fst_default_cache_gc` and `fst_default_cache_gc_limit` to
//! [`CacheOptions`](crate::cache::CacheOptions), and
//! `save_relabel_ipairs`/`save_relabel_opairs` to arguments of the relabel
//! tools. In sicada those are `Default` impls, so a caller who wants something
//! else passes it in rather than reaching for process-wide state that another
//! library in the same binary can flip underneath them. The seventh,
//! `fst_verify_properties`, becomes the `test` argument of
//! [`Fst::properties`].

use crate::AtomicRc;
use crate::arc::Arc;
use crate::error::OpenFstError;
use crate::fst_header::{FstHeader, flags};
use crate::properties::K_EXPANDED;
use crate::symbol_table::SymbolTable;

use std::fmt;
use std::fs::File;
use std::io::{BufReader, Read, Write};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

pub const FST_MAGIC_NUMBER: i32 = 2125659606;
pub const NO_LABEL: i32 = -1;
pub const NO_STATE_ID: i32 = -1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatchType {
    Input = 1,
    Output = 2,
    Both = 3,
    None = 4,
    Unknown = 5,
}

/// Whether a file should be read or memory mapped.
///
/// Advisory either way: plenty of conditions stop a file being mappable, and
/// reading is the fallback.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileReadMode {
    Read,
    Map,
}

impl FileReadMode {
    /// The name used on the command line and in `fst_read_mode`.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Read => "read",
            Self::Map => "map",
        }
    }

    /// Parses a mode name.
    ///
    /// SICADA-DIVERGE: upstream's `FstReadOptions::ReadMode` logs an error and
    /// returns `READ` for anything it does not recognize, so a typo silently
    /// changes behaviour. Returning `None` leaves that call to the caller.
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "read" => Some(Self::Read),
            "map" => Some(Self::Map),
            _ => None,
        }
    }
}

/// How to read an FST.
#[derive(Debug, Clone)]
pub struct FstReadOptions {
    /// Where the FST is being read from, for error messages.
    pub source: String,
    /// A header the caller has already read; if set, none is read from the
    /// stream, which must then be positioned just past it.
    pub header: Option<FstHeader>,
    /// Input symbols to use in place of the file's, which are still read past.
    pub isymbols: Option<AtomicRc<SymbolTable>>,
    /// Output symbols to use in place of the file's, which are still read past.
    pub osymbols: Option<AtomicRc<SymbolTable>>,
    /// Read or map the file, where the implementation can choose.
    pub mode: FileReadMode,
    /// Keep the file's input symbols, if it has any.
    pub read_isymbols: bool,
    /// Keep the file's output symbols, if it has any.
    pub read_osymbols: bool,
    /// Run the type's own sanity check on what was read, where it has one.
    pub verify: bool,
}

impl Default for FstReadOptions {
    fn default() -> Self {
        Self {
            source: "<unspecified>".to_string(),
            header: None,
            isymbols: None,
            osymbols: None,
            mode: FileReadMode::Read,
            read_isymbols: true,
            read_osymbols: true,
            verify: true,
        }
    }
}

impl FstReadOptions {
    /// Options for reading from `source`, with everything else left at its
    /// default.
    pub fn new<S: Into<String>>(source: S) -> Self {
        Self {
            source: source.into(),
            ..Default::default()
        }
    }

    /// Sets the read mode.
    pub fn mode(mut self, mode: FileReadMode) -> Self {
        self.mode = mode;
        self
    }

    /// Sets whether the file's symbol tables are kept, on both sides.
    pub fn read_symbols(mut self, read: bool) -> Self {
        self.read_isymbols = read;
        self.read_osymbols = read;
        self
    }

    /// Supplies a header the caller has already consumed from the stream.
    pub fn with_header(mut self, header: FstHeader) -> Self {
        self.header = Some(header);
        self
    }
}

/// The text upstream's `FstReadOptions::DebugString` produces.
impl fmt::Display for FstReadOptions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let set_or_null = |present: bool| if present { "set" } else { "null" };
        write!(
            f,
            "source: \"{}\" mode: \"{}\" read_isymbols: \"{}\" \
             read_osymbols: \"{}\" header: \"{}\" isymbols: \"{}\" \
             osymbols: \"{}\" verify: \"{}\"",
            self.source,
            self.mode.as_str().to_uppercase(),
            self.read_isymbols,
            self.read_osymbols,
            set_or_null(self.header.is_some()),
            set_or_null(self.isymbols.is_some()),
            set_or_null(self.osymbols.is_some()),
            self.verify
        )
    }
}

#[derive(Debug, Clone)]
pub struct FstWriteOptions {
    pub source: String,
    pub write_header: bool,
    pub write_isymbols: bool,
    pub write_osymbols: bool,
    pub align: bool,
    pub stream_write: bool,
}

impl Default for FstWriteOptions {
    fn default() -> Self {
        Self {
            source: "<unspecified>".to_string(),
            write_header: true,
            write_isymbols: true,
            write_osymbols: true,
            align: false,
            stream_write: false,
        }
    }
}

/// An FST's cache of property bits.
///
/// Port of the `properties_` member of upstream's `FstImplBase`, together with
/// the `UpdateProperties` that guards it. Property bits come from two places:
/// the operations that build an FST maintain them as they go (see
/// [`properties`](crate::properties)), and
/// [`compute_properties`](crate::algorithms::test_properties::compute_properties)
/// settles the rest by scanning. This is where the two meet.
#[derive(Debug, Default)]
pub struct PropertyCache(AtomicU64);

impl Clone for PropertyCache {
    fn clone(&self) -> Self {
        Self::new(self.get())
    }
}

impl PartialEq for PropertyCache {
    fn eq(&self, other: &Self) -> bool {
        self.get() == other.get()
    }
}

impl Eq for PropertyCache {}

impl PropertyCache {
    /// Creates a cache holding `props`.
    #[inline]
    pub fn new(props: u64) -> Self {
        Self(AtomicU64::new(props))
    }

    /// Every bit held, whether known or not.
    ///
    /// Relaxed throughout: the bits are a cache, never a way of ordering
    /// anything else.
    #[inline]
    pub fn get(&self) -> u64 {
        self.0.load(Ordering::Relaxed)
    }

    /// The bits held that `mask` asks for.
    #[inline]
    pub fn get_masked(&self, mask: u64) -> u64 {
        self.get() & mask
    }

    /// Replaces every bit.
    #[inline]
    pub fn set(&mut self, props: u64) {
        *self.0.get_mut() = props;
    }

    /// Replaces the bits with the result of `f`, for the incremental rules in
    /// [`properties`](crate::properties).
    #[inline]
    pub fn modify(&mut self, f: impl FnOnce(u64) -> u64) {
        let props = f(self.get());
        self.set(props);
    }

    /// Marks the FST as being in error.
    ///
    /// [`K_ERROR`](crate::properties::K_ERROR) is a binary property, so
    /// [`discover`](Self::discover), which only takes bits that were unsettled,
    /// can never set it. It is also the one bit that is only ever set and never
    /// cleared, which makes setting it through a shared reference safe to do.
    #[inline]
    pub fn mark_error(&self) {
        self.0
            .fetch_or(crate::properties::K_ERROR, Ordering::Relaxed);
    }

    /// Records what a scan settled, for the bits that were not settled before.
    ///
    /// Port of `FstImplBase::UpdateProperties`. Only newly settled bits are
    /// taken. Upstream explains why: an FST whose cache was set wrongly would
    /// otherwise end up holding a bit and its opposite at once, which is worse
    /// than holding one wrong bit, and plenty of code sets these wrongly.
    /// Correcting a wrong bit is [`set`](Self::set)'s job, not this one's.
    ///
    /// Takes `&self` because it runs from `properties(mask, /*test=*/true)`,
    /// which cannot ask for exclusive access.
    pub fn discover(&self, props: u64, mask: u64) {
        let known = crate::properties::internal::known_properties(self.get() & mask);
        let discovered = props & mask & !known;
        if discovered != 0 {
            self.0.fetch_or(discovered, Ordering::Relaxed);
        }
    }
}

/// A generic FST, templated on the arc definition.
/// All algorithm operations (`ops`) are generic over this trait.
pub trait Fst<A: Arc> {
    /// An iterator over the states of the FST.
    type StateIter<'a>: Iterator<Item = A::StateId>
    where
        Self: 'a;

    /// An iterator over the outgoing arcs of a state.
    /// `Clone` is required to allow multi-pass algorithms (e.g., matchers in Compose)
    /// to save and restore iteration positions effortlessly without C++ `Seek` or `Reset`.
    type ArcIter<'a>: Iterator<Item = A> + Clone
    where
        Self: 'a;

    /// Returns the initial state ID, or `None` if the FST is empty.
    fn start(&self) -> Option<A::StateId>;

    /// Returns the final weight of the given state.
    /// If the state is not final, this must return `Weight::zero()`.
    fn final_weight(&self, state: A::StateId) -> A::Weight;

    /// Returns the number of arcs leaving the given state.
    fn num_arcs(&self, state: A::StateId) -> usize;

    /// Returns the number of input epsilon arcs leaving the given state.
    fn num_input_epsilons(&self, state: A::StateId) -> usize;

    /// Returns the number of output epsilon arcs leaving the given state.
    fn num_output_epsilons(&self, state: A::StateId) -> usize;

    /// Returns the number of states if it is finite and can be computed in O(1) time.
    /// Otherwise returns `None`.
    fn num_states_if_known(&self) -> Option<usize>;

    /// Property bits.
    /// If `test` is false, returns stored properties bits (some possibly unknown).
    /// If `test` is true, computes the properties if they are unknown.
    fn properties(&self, mask: u64, test: bool) -> u64;

    /// Returns the name of the FST type (e.g., "vector", "const").
    fn fst_type(&self) -> &str;

    /// Returns the input label symbol table, if any.
    /// Using `Arc` allows cheap sharing across FST operations.
    fn input_symbols(&self) -> Option<AtomicRc<SymbolTable>>;

    /// Returns the output label symbol table, if any.
    fn output_symbols(&self) -> Option<AtomicRc<SymbolTable>>;

    /// Returns an iterator over all state IDs in the FST.
    fn states<'a>(&'a self) -> Self::StateIter<'a>;

    /// Returns an iterator over the outgoing arcs of the given state.
    fn arcs<'a>(&'a self, state: A::StateId) -> Self::ArcIter<'a>;

    /// Computes the exact number of states in the FST.
    /// If the state count is known in O(1) (i.e. `num_states_if_known` returns Some),
    /// it uses that. Otherwise, it iterates through all states in O(V) time.
    fn count_states(&self) -> usize {
        if let Some(n) = self.num_states_if_known() {
            n
        } else {
            self.states().count()
        }
    }

    /// Computes the exact number of arcs in the FST in O(V) time.
    fn count_arcs(&self) -> usize {
        self.states().map(|state| self.num_arcs(state)).sum()
    }
}

/// An FST whose total number of states is strictly known and reachable in O(1).
///
/// Implementations must return `Some(self.num_states())` from
/// [`Fst::num_states_if_known`]. Upstream gets that for free, because
/// `ExpandedFst` overrides `NumStatesIfKnown` in the base class, but the two are
/// separate methods here, so it is a rule rather than a consequence. Everything
/// that sizes a buffer from `num_states_if_known` and then indexes it by a state
/// ID depends on it; `dfs_visit` is one.
pub trait ExpandedFst<A: Arc>: Fst<A> {
    /// Returns the total number of states in the FST.
    fn num_states(&self) -> usize;
}

/// An FST that can be mutated (states and arcs added/removed).
pub trait MutableFst<A: Arc>: ExpandedFst<A> {
    /// Sets the initial state.
    ///
    /// Passing [`ArcStateId::no_state`](crate::arc::ArcStateId::no_state) clears it, so that
    /// [`Fst::start`] reports `None`.
    fn set_start(&mut self, state: A::StateId);

    /// Sets the final weight of a given state.
    /// Setting it to `Weight::zero()` effectively marks the state as non-final.
    fn set_final(&mut self, state: A::StateId, weight: A::Weight);

    /// Explicitly updates the properties mask.
    fn set_properties(&mut self, props: u64, mask: u64);

    /// Adds a new state to the FST and returns its ID.
    fn add_state(&mut self) -> A::StateId;

    /// Adds `n` new states to the FST.
    fn add_states(&mut self, n: usize);

    /// Adds an outgoing arc to a given state.
    fn add_arc(&mut self, state: A::StateId, arc: A);

    /// The arcs leaving `state`, to be rearranged in place.
    ///
    /// Empty for a state that does not exist. Port of upstream's
    /// `MutableArcIterator`, which exists for the same reason: rewriting a
    /// state's arcs by deleting them and adding them back costs a pass over
    /// them per step, and makes every `add_arc` re-derive the property bits
    /// one arc at a time.
    ///
    /// The property bits are *not* updated. Rearranging arcs cannot change
    /// them, which is why a sort can use this and leave them alone; a caller
    /// that rewrites a label, a weight or a destination has to set them
    /// afterwards.
    fn arcs_mut(&mut self, state: A::StateId) -> &mut [A];

    /// Deletes `n` outgoing arcs from a given state.
    fn delete_arcs_n(&mut self, state: A::StateId, n: usize);

    /// Deletes all outgoing arcs from a given state.
    fn delete_arcs(&mut self, state: A::StateId);

    /// Deletes all states and arcs, leaving the FST empty.
    fn delete_all_states(&mut self);

    /// Deletes specific states.
    /// Note: This renumbers the remaining states and invalidates existing Arc nextstates.
    fn delete_states(&mut self, states: &[A::StateId]);

    /// Hints the underlying allocation to reserve space for `n` total states.
    fn reserve_states(&mut self, n: usize);

    /// Hints the underlying allocation to reserve space for `n` arcs on a specific state.
    fn reserve_arcs(&mut self, state: A::StateId, n: usize);

    /// Attaches an input symbol table.
    fn set_input_symbols(&mut self, syms: Option<AtomicRc<SymbolTable>>);

    /// Attaches an output symbol table.
    fn set_output_symbols(&mut self, syms: Option<AtomicRc<SymbolTable>>);

    /// The input symbol table, to be changed in place.
    ///
    /// Returns `None` when there is no table. The table is copied first if it
    /// is shared, so changing it here never reaches another FST holding the
    /// same one.
    fn mutable_input_symbols(&mut self) -> Option<&mut SymbolTable>;

    /// The output symbol table, to be changed in place. See
    /// [`mutable_input_symbols`](Self::mutable_input_symbols).
    fn mutable_output_symbols(&mut self) -> Option<&mut SymbolTable>;

    /// Rewrites every arc leaving `state`.
    ///
    /// SICADA-DIVERGE: upstream exposes a `MutableArcIterator` whose `SetValue`
    /// writes back through the iterator, and warns that adding or removing arcs
    /// while one is open invalidates it. Handing the arcs to a closure says the
    /// same thing with no way to get it wrong, and lets the FST recompute its
    /// property bits once at the end rather than per arc.
    fn mutate_arcs<F>(&mut self, state: A::StateId, mutator: F)
    where
        F: FnMut(&mut A);
}

/// An FST whose arcs for any given state are stored contiguously in memory.
/// An FST whose arcs for a state sit next to each other in memory.
///
/// Opt-in: an FST that stores its arcs contiguously says so by implementing
/// this, and an algorithm that wants random access over them asks for it in its
/// bounds. `VectorFst` and `ConstFst` do; a delayed FST that produces arcs one
/// at a time does not.
///
/// This is the gap upstream's `arc-range.h` fills.
pub trait ContiguousArcsFst<A: Arc>: Fst<A> {
    /// Returns a contiguous slice of arcs leaving the given state.
    fn arcs_slice(&self, state: A::StateId) -> &[A];
}

/// Returns the total number of states across several FSTs, counting them where
/// they do not know.
///
/// SICADA-DIVERGE: upstream takes a `std::vector<const Fst<Arc>*>`, so the FSTs
/// may be of different types. `Fst` has generic associated types and so is not
/// dyn-compatible; a caller with mixed types sums `count_states` itself.
#[inline]
pub fn count_states_slice<A: Arc, F: Fst<A>>(fsts: &[&F]) -> usize {
    fsts.iter().map(|fst| fst.count_states()).sum()
}

/// Returns the number of states in an FST, counting them if it does not know.
///
/// Port of upstream's `CountStates`. Prefer the [`Fst::count_states`] method.
#[inline(always)]
pub fn count_states<A: Arc, F: Fst<A>>(fst: &F) -> usize {
    fst.count_states()
}

/// Returns the number of arcs in an FST.
///
/// Port of upstream's `CountArcs`. Prefer the [`Fst::count_arcs`] method.
#[inline(always)]
pub fn count_arcs<A: Arc, F: Fst<A>>(fst: &F) -> usize {
    fst.count_arcs()
}

/// A header together with the symbol tables that followed it in the file.
#[derive(Debug, Clone)]
pub struct FstHeaderWithSymbols {
    pub header: FstHeader,
    pub isymbols: Option<AtomicRc<SymbolTable>>,
    pub osymbols: Option<AtomicRc<SymbolTable>>,
}

/// Reads the header and the symbol tables behind it, leaving the stream on the
/// FST's own data.
///
/// Port of `FstImpl::ReadHeader`. Every concrete FST reader starts here: the
/// symbol tables sit between the header and the states, so a reader that skips
/// them reads the states out of the middle of a symbol table.
///
/// `fst_type` is the type the caller is prepared to read; a file claiming
/// anything else is an error, as is one written by a version older than
/// `min_version`.
pub fn read_fst_header<A: Arc, R: Read>(
    reader: &mut R,
    opts: &FstReadOptions,
    fst_type: &str,
    min_version: i32,
) -> Result<FstHeaderWithSymbols, OpenFstError> {
    let header = match &opts.header {
        Some(header) => header.clone(),
        None => FstHeader::read(&mut *reader)?,
    };

    if header.fst_type != fst_type {
        return Err(OpenFstError::InvalidFstHeader(format!(
            "{}: FST not of type '{}', found '{}'",
            opts.source, fst_type, header.fst_type
        )));
    }
    let arc_type = A::type_name();
    if header.arc_type != arc_type.as_str() {
        return Err(OpenFstError::InvalidFstHeader(format!(
            "{}: arc not of type '{}', found '{}'",
            opts.source,
            arc_type.as_str(),
            header.arc_type
        )));
    }
    if header.version < min_version {
        return Err(OpenFstError::InvalidFstHeader(format!(
            "{}: obsolete {} FST version {}, min_version={}",
            opts.source, fst_type, header.version, min_version
        )));
    }

    // Both tables are read past even when the caller does not want them, or
    // supplies its own: they are in the way of the states either way.
    let mut isymbols = None;
    if header.flags & flags::HAS_ISYMBOLS != 0 {
        let table = SymbolTable::read(&mut *reader)?;
        if opts.read_isymbols {
            isymbols = Some(AtomicRc::new(table));
        }
    }
    let mut osymbols = None;
    if header.flags & flags::HAS_OSYMBOLS != 0 {
        let table = SymbolTable::read(&mut *reader)?;
        if opts.read_osymbols {
            osymbols = Some(AtomicRc::new(table));
        }
    }
    if opts.isymbols.is_some() {
        isymbols = opts.isymbols.clone();
    }
    if opts.osymbols.is_some() {
        osymbols = opts.osymbols.clone();
    }

    Ok(FstHeaderWithSymbols {
        header,
        isymbols,
        osymbols,
    })
}

/// Writes the header and the symbol tables that belong behind it.
///
/// Port of `FstImpl::WriteHeader` and its static twin `WriteFstHeader`, which
/// differ upstream only in where they read the symbol tables from. `header`
/// supplies the FST's own fields; the flags are computed here from `opts` and
/// the tables actually being written, and the header as written is returned so
/// that a writer which has to come back and correct the state and arc counts
/// can rewrite the same bytes.
pub fn write_fst_header<W: Write>(
    writer: &mut W,
    opts: &FstWriteOptions,
    header: &FstHeader,
    isymbols: Option<&SymbolTable>,
    osymbols: Option<&SymbolTable>,
) -> Result<FstHeader, OpenFstError> {
    let isymbols = isymbols.filter(|_| opts.write_isymbols);
    let osymbols = osymbols.filter(|_| opts.write_osymbols);

    let mut header = header.clone();
    header.flags = 0;
    if isymbols.is_some() {
        header.flags |= flags::HAS_ISYMBOLS;
    }
    if osymbols.is_some() {
        header.flags |= flags::HAS_OSYMBOLS;
    }
    if opts.align {
        header.flags |= flags::IS_ALIGNED;
    }

    if opts.write_header {
        header.write(&mut *writer)?;
    }
    if let Some(table) = isymbols {
        table.write(&mut *writer)?;
    }
    if let Some(table) = osymbols {
        table.write(&mut *writer)?;
    }
    Ok(header)
}

/// An FST that can be read back from a stream.
pub trait ReadFst<A: Arc>: Sized {
    /// Reads one FST, starting at its header.
    ///
    /// Pass the header through `opts` if it has already been consumed.
    fn read_from_stream<R: Read>(
        reader: &mut R,
        opts: &FstReadOptions,
    ) -> Result<Self, OpenFstError>;
}

/// Reads an FST from a file.
pub fn read_fst_from_file<A, F>(
    path: impl AsRef<Path>,
    opts: &FstReadOptions,
) -> Result<F, OpenFstError>
where
    A: Arc,
    F: Fst<A> + ExpandedFst<A> + ReadFst<A>,
{
    let path = path.as_ref();
    let mut reader = BufReader::new(File::open(path)?);

    // Read the header here rather than leaving it to the implementation, so
    // that a file of the wrong shape is rejected before any of it is decoded,
    // then hand it on so it is not read twice.
    let header = FstHeader::read(&mut reader)?;

    if (header.properties & K_EXPANDED) == 0 {
        return Err(OpenFstError::InvalidFstHeader(format!(
            "Not an ExpandedFst (K_EXPANDED property is missing): {:?}",
            path
        )));
    }

    let arc_type = A::type_name();
    if header.arc_type != arc_type.as_str() {
        return Err(OpenFstError::InvalidFstHeader(format!(
            "Arc type mismatch. Expected '{}', got '{}'",
            arc_type.as_str(),
            header.arc_type
        )));
    }

    let opts = FstReadOptions {
        source: path.display().to_string(),
        header: Some(header),
        ..opts.clone()
    };
    F::read_from_stream(&mut reader, &opts)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arc::StdArc;
    use crate::weight::Weight;
    use crate::weights::float_weight::TropicalWeight;
    use std::io::Cursor;

    fn symbols(name: &str, first: &str) -> SymbolTable {
        let mut table = SymbolTable::new(name.to_string());
        table.add_symbol("<eps>", 0);
        table.add_symbol(first, 1);
        table
    }

    fn header() -> FstHeader {
        FstHeader {
            fst_type: "vector".to_string(),
            arc_type: StdArc::type_name().as_str().to_string(),
            version: 2,
            // Overwritten by `write_fst_header`; set to catch it not being.
            flags: 0xffff_ffff,
            properties: K_EXPANDED,
            start: 0,
            num_states: 3,
            num_arcs: 2,
        }
    }

    /// The flags say which tables follow, so writing and reading has to agree
    /// about them or the states are read out of the middle of a symbol table.
    #[test]
    fn a_header_and_its_symbols_round_trip() {
        let isymbols = symbols("input", "a");
        let osymbols = symbols("output", "x");

        let mut bytes = Vec::new();
        let written = write_fst_header(
            &mut bytes,
            &FstWriteOptions::default(),
            &header(),
            Some(&isymbols),
            Some(&osymbols),
        )
        .unwrap();
        assert_eq!(written.flags, flags::HAS_ISYMBOLS | flags::HAS_OSYMBOLS);

        // Whatever follows the tables must be found intact.
        bytes.extend_from_slice(b"the states");

        let mut reader = Cursor::new(bytes);
        let read =
            read_fst_header::<StdArc, _>(&mut reader, &FstReadOptions::default(), "vector", 2)
                .unwrap();

        assert_eq!(read.header, written);
        assert_eq!(read.isymbols.unwrap().find_symbol(1), Some("a"));
        assert_eq!(read.osymbols.unwrap().find_symbol(1), Some("x"));
        let mut rest = Vec::new();
        reader.read_to_end(&mut rest).unwrap();
        assert_eq!(rest, b"the states");
    }

    /// Not wanting a table is not the same as it not being there: it still has
    /// to be read past.
    #[test]
    fn unwanted_symbols_are_still_read_past() {
        let mut bytes = Vec::new();
        write_fst_header(
            &mut bytes,
            &FstWriteOptions::default(),
            &header(),
            Some(&symbols("input", "a")),
            Some(&symbols("output", "x")),
        )
        .unwrap();
        bytes.extend_from_slice(b"the states");

        let mut reader = Cursor::new(bytes);
        let read = read_fst_header::<StdArc, _>(
            &mut reader,
            &FstReadOptions::default().read_symbols(false),
            "vector",
            2,
        )
        .unwrap();
        assert!(read.isymbols.is_none() && read.osymbols.is_none());
        let mut rest = Vec::new();
        reader.read_to_end(&mut rest).unwrap();
        assert_eq!(rest, b"the states");
    }

    /// Supplied tables replace the file's, which are still consumed.
    #[test]
    fn supplied_symbols_replace_the_files_own() {
        let mut bytes = Vec::new();
        write_fst_header(
            &mut bytes,
            &FstWriteOptions::default(),
            &header(),
            Some(&symbols("input", "a")),
            None,
        )
        .unwrap();
        bytes.extend_from_slice(b"the states");

        let opts = FstReadOptions {
            isymbols: Some(AtomicRc::new(symbols("supplied", "z"))),
            ..Default::default()
        };
        let mut reader = Cursor::new(bytes);
        let read = read_fst_header::<StdArc, _>(&mut reader, &opts, "vector", 2).unwrap();
        assert_eq!(read.isymbols.unwrap().find_symbol(1), Some("z"));
        let mut rest = Vec::new();
        reader.read_to_end(&mut rest).unwrap();
        assert_eq!(rest, b"the states");
    }

    #[test]
    fn a_header_supplied_by_the_caller_is_not_read_from_the_stream() {
        let mut bytes = Vec::new();
        let written = write_fst_header(
            &mut bytes,
            &FstWriteOptions::default(),
            &header(),
            Some(&symbols("input", "a")),
            None,
        )
        .unwrap();
        let header_len = bytes.len();
        bytes.extend_from_slice(b"the states");

        // Start reading past the header, as a caller that already consumed it
        // would leave the stream.
        let symbols_at = {
            let mut probe = Vec::new();
            written.write(&mut probe).unwrap();
            probe.len()
        };
        let mut reader = Cursor::new(bytes[symbols_at..].to_vec());
        let opts = FstReadOptions::default().with_header(written.clone());
        let read = read_fst_header::<StdArc, _>(&mut reader, &opts, "vector", 2).unwrap();

        assert_eq!(read.header, written);
        assert_eq!(read.isymbols.unwrap().find_symbol(1), Some("a"));
        let mut rest = Vec::new();
        reader.read_to_end(&mut rest).unwrap();
        assert_eq!(rest, b"the states");
        assert!(header_len > symbols_at);
    }

    #[test]
    fn a_mismatched_type_version_or_arc_is_refused() {
        let mut bytes = Vec::new();
        write_fst_header(
            &mut bytes,
            &FstWriteOptions::default(),
            &header(),
            None,
            None,
        )
        .unwrap();

        let opts = FstReadOptions::default();
        let read = |fst_type: &str, min_version| {
            read_fst_header::<StdArc, _>(
                &mut Cursor::new(bytes.clone()),
                &opts,
                fst_type,
                min_version,
            )
        };
        assert!(read("vector", 2).is_ok());
        assert!(read("const", 2).is_err(), "wrong FST type accepted");
        assert!(read("vector", 3).is_err(), "obsolete version accepted");

        // A file written for a different arc type must not be decoded as this
        // one, whatever the FST type says.
        let mut other = header();
        other.arc_type = "log".to_string();
        let mut bytes = Vec::new();
        write_fst_header(&mut bytes, &FstWriteOptions::default(), &other, None, None).unwrap();
        assert!(
            read_fst_header::<StdArc, _>(&mut Cursor::new(bytes), &opts, "vector", 2).is_err(),
            "wrong arc type accepted"
        );
    }

    #[test]
    fn writing_can_be_told_to_skip_the_header_or_a_table() {
        let opts = FstWriteOptions {
            write_header: false,
            write_osymbols: false,
            ..Default::default()
        };
        let mut bytes = Vec::new();
        let written = write_fst_header(
            &mut bytes,
            &opts,
            &header(),
            Some(&symbols("input", "a")),
            Some(&symbols("output", "x")),
        )
        .unwrap();
        // The flags describe what was written, so the skipped table is absent
        // from them too.
        assert_eq!(written.flags, flags::HAS_ISYMBOLS);

        let mut table_only = Vec::new();
        symbols("input", "a").write(&mut table_only).unwrap();
        assert_eq!(bytes, table_only);
    }

    #[test]
    fn alignment_is_recorded_in_the_flags() {
        let opts = FstWriteOptions {
            align: true,
            ..Default::default()
        };
        let mut bytes = Vec::new();
        let written = write_fst_header(&mut bytes, &opts, &header(), None, None).unwrap();
        assert_eq!(written.flags, flags::IS_ALIGNED);
    }

    #[test]
    fn read_modes_round_trip_through_their_names() {
        for mode in [FileReadMode::Read, FileReadMode::Map] {
            assert_eq!(FileReadMode::from_name(mode.as_str()), Some(mode));
        }
        assert_eq!(FileReadMode::from_name("READ"), None);
        assert_eq!(FileReadMode::from_name("mmap"), None);
    }

    /// The text upstream's `DebugString` methods produce, so log output from
    /// the two libraries can be compared directly.
    #[test]
    fn options_and_headers_print_the_way_openfst_prints_them() {
        assert_eq!(
            FstReadOptions::new("x.fst").to_string(),
            "source: \"x.fst\" mode: \"READ\" read_isymbols: \"true\" \
             read_osymbols: \"true\" header: \"null\" isymbols: \"null\" \
             osymbols: \"null\" verify: \"true\""
        );
        let opts = FstReadOptions {
            isymbols: Some(AtomicRc::new(symbols("input", "a"))),
            mode: FileReadMode::Map,
            read_osymbols: false,
            verify: false,
            ..FstReadOptions::new("y.fst")
        };
        assert_eq!(
            opts.to_string(),
            "source: \"y.fst\" mode: \"MAP\" read_isymbols: \"true\" \
             read_osymbols: \"false\" header: \"null\" isymbols: \"set\" \
             osymbols: \"null\" verify: \"false\""
        );

        let mut header = header();
        header.flags = flags::HAS_ISYMBOLS;
        assert_eq!(
            header.to_string(),
            "fsttype: \"vector\" arctype: \"standard\" version: \"2\" \
             flags: \"1\" properties: \"1\" start: \"0\" numstates: \"3\" \
             numarcs: \"2\""
        );
    }

    /// An `ExpandedFst` has to answer the same number twice over: upstream ties
    /// the two together in a base class, so nothing can drift, while here each
    /// implementation writes both by hand.
    #[test]
    fn expanded_fsts_agree_with_themselves_about_their_size() {
        use crate::fsts::const_fst::ConstFst;
        use crate::fsts::vector_fst::VectorFst;

        fn check<A: Arc, F: ExpandedFst<A>>(fst: &F, expected: usize) {
            assert_eq!(fst.num_states(), expected);
            assert_eq!(fst.num_states_if_known(), Some(expected));
            assert_eq!(fst.count_states(), expected);
        }

        let mut vector = VectorFst::<StdArc>::new();
        for _ in 0..4 {
            vector.add_state();
        }
        vector.set_start(0);
        vector.add_arc(0, StdArc::new(1, 1, TropicalWeight::one(), 1));
        vector.add_arc(0, StdArc::new(2, 2, TropicalWeight::one(), 2));
        vector.add_arc(1, StdArc::new(3, 3, TropicalWeight::one(), 3));
        check(&vector, 4);

        let constant = ConstFst::<StdArc, u32>::from_fst(&vector).unwrap();
        check(&constant, 4);

        assert_eq!(vector.count_arcs(), 3);
        assert_eq!(constant.count_arcs(), 3);
    }

    /// An FST that does not know its size gets counted rather than guessed at.
    #[test]
    fn an_fst_that_does_not_know_its_size_is_counted() {
        struct Unsized(usize);

        impl Fst<StdArc> for Unsized {
            type StateIter<'a> = std::ops::Range<i32>;
            type ArcIter<'a> = std::iter::Empty<StdArc>;

            fn start(&self) -> Option<i32> {
                Some(0)
            }

            fn final_weight(&self, _state: i32) -> TropicalWeight {
                TropicalWeight::zero()
            }

            fn num_arcs(&self, _state: i32) -> usize {
                0
            }

            fn num_input_epsilons(&self, _state: i32) -> usize {
                0
            }

            fn num_output_epsilons(&self, _state: i32) -> usize {
                0
            }

            fn num_states_if_known(&self) -> Option<usize> {
                None
            }

            fn properties(&self, _mask: u64, _test: bool) -> u64 {
                0
            }

            fn fst_type(&self) -> &str {
                "unsized"
            }

            fn input_symbols(&self) -> Option<AtomicRc<SymbolTable>> {
                None
            }

            fn output_symbols(&self) -> Option<AtomicRc<SymbolTable>> {
                None
            }

            fn states<'a>(&'a self) -> Self::StateIter<'a> {
                0..self.0 as i32
            }

            fn arcs<'a>(&'a self, _state: i32) -> Self::ArcIter<'a> {
                std::iter::empty()
            }
        }

        let fst = Unsized(7);
        assert_eq!(fst.num_states_if_known(), None);
        assert_eq!(fst.count_states(), 7);
        assert_eq!(count_states(&fst), 7);
        assert_eq!(count_states_slice(&[&fst, &fst]), 14);
    }
}

#[cfg(test)]
mod arc_range_tests {
    //! What upstream's `arc-range.h` exists to provide, and how it is reached
    //! here instead.

    use super::*;
    use crate::arc::StdArc;
    use crate::fsts::vector_fst::StdVectorFst;
    use crate::weight::Weight;
    use crate::weights::float_weight::TropicalWeight;

    fn fst() -> StdVectorFst {
        let mut fst = StdVectorFst::new();
        for _ in 0..2 {
            fst.add_state();
        }
        fst.set_start(0);
        fst.set_final(1, TropicalWeight::one());
        for label in [7, 42, 9] {
            fst.add_arc(0, StdArc::new(label, label, TropicalWeight::one(), 1));
        }
        fst
    }

    /// The example from the header of `arc-range.h`: a range-based loop and a
    /// search for an arc by label. Both are what `arcs()` already returns.
    #[test]
    fn arcs_are_iterable_and_searchable_without_a_wrapper() {
        let fst = fst();

        let mut labels = Vec::new();
        for arc in fst.arcs(0) {
            labels.push(arc.olabel());
        }
        assert_eq!(labels, vec![7, 42, 9]);

        let found = fst.arcs(0).find(|arc| arc.olabel() == 42);
        assert!(found.is_some());
        assert_eq!(fst.arcs(0).filter(|arc| arc.olabel() > 8).count(), 2);
    }

    /// The other half: an FST that stores its arcs contiguously hands out a
    /// slice, which is random access without a seek per element.
    #[test]
    fn contiguous_arcs_are_a_slice() {
        let fst = fst();
        let arcs = fst.arcs_slice(0);
        assert_eq!(arcs.len(), 3);
        assert_eq!(arcs[1].olabel(), 42);
        assert_eq!(arcs.last().unwrap().olabel(), 9);
        assert!(arcs.binary_search_by_key(&7, |arc| arc.olabel()).is_ok());
    }
}
