//! Folding labels and weights into a single label, and unfolding them again.
//!
//! Port of OpenFst's `encode.h`. Determinization and minimization are defined
//! for unweighted acceptors; encoding lets them be run on a weighted
//! transducer. Each distinct (input label, output label, weight) triple is given
//! a label of its own, so the FST becomes an unweighted acceptor over those
//! labels, and decoding puts the triples back.
//!
//! The table is shared between the encoder and the decoder, and is filled in as
//! encoding proceeds, so `encode → determinize → decode` works even though the
//! table is incomplete when the decoder is built.

use std::cell::RefCell;
use std::hash::Hash;
use std::io::{Read, Write};
use std::rc::Rc;

use crate::AtomicRc;
use crate::algorithms::arc_map::{ArcMapper, MapFinalAction, MapSymbolsAction, arc_map};
use crate::algorithms::rmfinalepsilon::rm_final_epsilon;
use crate::arc::{Arc, ArcLabel, ArcStateId};
use crate::data_structures::bi_table::CompactHashBiTable;
use crate::error::OpenFstError;
use crate::fst::{Fst, MutableFst};
use crate::fst_type::ArcType;
use crate::properties::{
    K_ACCEPTOR, K_ADD_SUPER_FINAL_PROPERTIES, K_ERROR, K_FST_PROPERTIES, K_I_DETERMINISTIC,
    K_I_LABEL_INVARIANT_PROPERTIES, K_O_LABEL_INVARIANT_PROPERTIES, K_RM_SUPER_FINAL_PROPERTIES,
    K_UNWEIGHTED, K_UNWEIGHTED_CYCLES, K_WEIGHT_INVARIANT_PROPERTIES,
};
use crate::symbol_table::SymbolTable;
use crate::utils::io::{FstScalar, read_scalar, read_string, write_scalar, write_string};
use crate::weight::{Weight, WeightIo};

/// Fold the output label into the encoded label, making the result an acceptor.
pub const ENCODE_LABELS: u8 = 0x01;
/// Fold the weight into the encoded label, making the result unweighted.
pub const ENCODE_WEIGHTS: u8 = 0x02;
/// Both of the above.
pub const ENCODE_FLAGS: u8 = ENCODE_LABELS | ENCODE_WEIGHTS;

/// Set in the stored flags when the table carries an input symbol table.
const ENCODE_HAS_ISYMBOLS: u8 = 0x04;
/// Set in the stored flags when the table carries an output symbol table.
const ENCODE_HAS_OSYMBOLS: u8 = 0x08;

/// Identifies a stream as an encode table, and its endianness.
pub const ENCODE_MAGIC_NUMBER: i32 = 2128178506;
/// The magic number of the pre-2019 encode table format, still readable.
const ENCODE_DEPRECATED_MAGIC_NUMBER: i32 = 2129983209;

/// Which direction a mapper runs in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncodeType {
    /// Triples to labels.
    Encode,
    /// Labels back to triples.
    Decode,
}

/// What an encoded label stands for.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Triple<L, W> {
    /// The original input label.
    pub ilabel: L,
    /// The original output label, or epsilon when labels are not encoded.
    pub olabel: L,
    /// The original weight, or [`Weight::one`] when weights are not encoded.
    pub weight: W,
}

impl<L: ArcLabel, W: Weight> Triple<L, W> {
    /// The triple an arc stands for under `flags`.
    ///
    /// What the flags do not cover is left at its identity, so that two arcs
    /// differing only in a field that is not being encoded get the same label.
    fn from_arc<A: Arc<Label = L, Weight = W>>(arc: &A, flags: u8) -> Self {
        Self {
            ilabel: arc.ilabel(),
            olabel: if flags & ENCODE_LABELS != 0 {
                arc.olabel()
            } else {
                L::epsilon()
            },
            weight: if flags & ENCODE_WEIGHTS != 0 {
                arc.weight().clone()
            } else {
                W::one()
            },
        }
    }
}

/// The header of an encode table on disk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncodeTableHeader {
    /// The arc type the table was written for.
    pub arc_type: String,
    /// The encode flags, including the internal symbol-table bits.
    pub flags: u8,
    /// How many triples follow.
    pub size: u64,
}

impl EncodeTableHeader {
    /// Reads a header, accepting both the current and the deprecated format.
    pub fn read<R: Read>(reader: &mut R) -> Result<Self, OpenFstError> {
        let magic: i32 = read_scalar(reader)?;
        match magic {
            ENCODE_MAGIC_NUMBER => Ok(Self {
                arc_type: read_string(reader)?,
                flags: read_scalar(reader)?,
                size: read_scalar(reader)?,
            }),
            ENCODE_DEPRECATED_MAGIC_NUMBER => {
                // The old format had no arc type, 32-bit flags and a signed
                // size.
                let flags: u32 = read_scalar(reader)?;
                let size: i64 = read_scalar(reader)?;
                Ok(Self {
                    arc_type: String::new(),
                    flags: flags as u8,
                    size: size as u64,
                })
            }
            _ => Err(OpenFstError::InvalidFstHeader(format!(
                "EncodeTableHeader::read: bad magic number {magic}"
            ))),
        }
    }

    /// Writes a header in the current format.
    pub fn write<W: Write>(&self, writer: &mut W) -> Result<(), OpenFstError> {
        write_scalar(writer, ENCODE_MAGIC_NUMBER)?;
        write_string(writer, &self.arc_type)?;
        write_scalar(writer, self.flags)?;
        write_scalar(writer, self.size)?;
        Ok(())
    }
}

/// The bijection between triples and encoded labels.
///
/// SICADA-OPT: upstream stores `unique_ptr<const Triple>` in a vector and keys
/// a hash map on the raw pointers, so that the keys survive a rehash, at the
/// cost of one heap allocation per distinct triple and a pointer chase on every
/// lookup. The same problem was already solved by `bi-table.h`, which upstream
/// does not use here: [`CompactHashBiTable`] keeps the entries in one vector and
/// the hash table holds indices into it.
pub struct EncodeTable<L, W> {
    flags: u8,
    triples: CompactHashBiTable<usize, Triple<L, W>>,
    isymbols: Option<AtomicRc<SymbolTable>>,
    osymbols: Option<AtomicRc<SymbolTable>>,
}

impl<L, W> EncodeTable<L, W>
where
    L: ArcLabel,
    W: Weight + Hash + Eq,
{
    /// An empty table encoding what `flags` asks for.
    pub fn new(flags: u8) -> Self {
        Self {
            flags,
            triples: CompactHashBiTable::new(1024),
            isymbols: None,
            osymbols: None,
        }
    }

    /// The label standing for `arc`, assigning one if this is the first time it
    /// has been seen.
    pub fn encode<A: Arc<Label = L, Weight = W>>(&mut self, arc: &A) -> L {
        // Encoding the weight of a hallucinated superfinal transition could
        // collide with a true epsilon arc, so it is given `no_label` on both
        // sides instead, which is a triple no real arc can produce.
        let triple =
            if arc.nextstate() == A::StateId::no_state() && self.flags & ENCODE_WEIGHTS != 0 {
                Triple {
                    ilabel: L::no_label(),
                    olabel: L::no_label(),
                    weight: arc.weight().clone(),
                }
            } else {
                Triple::from_arc(arc, self.flags)
            };
        self.encode_triple(triple)
    }

    /// The label for a triple, assigning one if it is new.
    ///
    /// Labels count from 1, since 0 is epsilon.
    fn encode_triple(&mut self, triple: Triple<L, W>) -> L {
        let id = self
            .triples
            .find_id(&triple, true)
            .expect("find_id inserts when asked to");
        L::from_i64(id as i64 + 1).unwrap_or_else(L::no_label)
    }

    /// What `label` stands for, or `None` if no such label was ever assigned.
    pub fn decode(&self, label: L) -> Option<&Triple<L, W>> {
        let index = label.to_i64()?.checked_sub(1)?;
        self.triples.find_entry(usize::try_from(index).ok()?)
    }

    /// How many labels have been assigned.
    pub fn len(&self) -> usize {
        self.triples.size()
    }

    /// Whether no label has been assigned.
    pub fn is_empty(&self) -> bool {
        self.triples.size() == 0
    }

    /// The encode flags, without the internal symbol-table bits.
    pub fn flags(&self) -> u8 {
        self.flags & ENCODE_FLAGS
    }

    /// The input symbol table the encoded FST had.
    pub fn input_symbols(&self) -> Option<AtomicRc<SymbolTable>> {
        self.isymbols.clone()
    }

    /// The output symbol table the encoded FST had.
    pub fn output_symbols(&self) -> Option<AtomicRc<SymbolTable>> {
        self.osymbols.clone()
    }

    /// Remembers the input symbol table, so that decoding can put it back.
    pub fn set_input_symbols(&mut self, syms: Option<AtomicRc<SymbolTable>>) {
        match syms {
            Some(syms) => {
                self.isymbols = Some(syms);
                self.flags |= ENCODE_HAS_ISYMBOLS;
            }
            None => {
                self.isymbols = None;
                self.flags &= !ENCODE_HAS_ISYMBOLS;
            }
        }
    }

    /// Remembers the output symbol table, so that decoding can put it back.
    pub fn set_output_symbols(&mut self, syms: Option<AtomicRc<SymbolTable>>) {
        match syms {
            Some(syms) => {
                self.osymbols = Some(syms);
                self.flags |= ENCODE_HAS_OSYMBOLS;
            }
            None => {
                self.osymbols = None;
                self.flags &= !ENCODE_HAS_OSYMBOLS;
            }
        }
    }
}

impl<L, W> EncodeTable<L, W>
where
    L: ArcLabel + FstScalar,
    W: Weight + WeightIo + Hash + Eq,
{
    /// Reads a table in OpenFst's encode-table format.
    pub fn read<R: Read>(reader: &mut R) -> Result<Self, OpenFstError> {
        let header = EncodeTableHeader::read(reader)?;
        let mut table = Self::new(header.flags);
        // SICADA-DIVERGE: upstream loops `size` times reading triples, so a
        // corrupt size makes it read (and allocate) until the stream runs out.
        // The reader here fails on a short stream instead, and nothing is
        // reserved up front.
        for _ in 0..header.size {
            let triple = Triple {
                ilabel: read_scalar(reader)?,
                olabel: read_scalar(reader)?,
                weight: W::read(reader)?,
            };
            table.encode_triple(triple);
        }
        if header.flags & ENCODE_HAS_ISYMBOLS != 0 {
            table.isymbols = Some(AtomicRc::new(SymbolTable::read(reader)?));
        }
        if header.flags & ENCODE_HAS_OSYMBOLS != 0 {
            table.osymbols = Some(AtomicRc::new(SymbolTable::read(reader)?));
        }
        Ok(table)
    }

    /// Writes the table in OpenFst's encode-table format.
    pub fn write<Wr: Write>(&self, writer: &mut Wr, arc_type: ArcType) -> Result<(), OpenFstError> {
        EncodeTableHeader {
            arc_type: arc_type.to_string(),
            // The stored flags, symbol-table bits included.
            flags: self.flags,
            size: self.len() as u64,
        }
        .write(writer)?;
        for index in 0..self.len() {
            let triple = self.triples.find_entry(index).expect("index is in range");
            write_scalar(writer, triple.ilabel)?;
            write_scalar(writer, triple.olabel)?;
            triple.weight.write(writer)?;
        }
        if self.flags & ENCODE_HAS_ISYMBOLS != 0
            && let Some(syms) = &self.isymbols
        {
            syms.write(writer)?;
        }
        if self.flags & ENCODE_HAS_OSYMBOLS != 0
            && let Some(syms) = &self.osymbols
        {
            syms.write(writer)?;
        }
        Ok(())
    }
}

/// Rewrites arcs into encoded labels, or back again.
///
/// The table is shared: a decoder built from an encoder sees every label the
/// encoder assigns, including ones assigned after the decoder was made. That is
/// what lets `encode → determinize → decode` work on a table that is still
/// being filled in.
pub struct EncodeMapper<A: Arc> {
    flags: u8,
    encode_type: EncodeType,
    table: Rc<RefCell<EncodeTable<A::Label, A::Weight>>>,
    error: bool,
}

impl<A: Arc> EncodeMapper<A>
where
    A::Weight: Hash + Eq,
{
    /// An encoder with an empty table.
    ///
    /// `flags` is some combination of [`ENCODE_LABELS`] and [`ENCODE_WEIGHTS`].
    pub fn new(flags: u8) -> Self {
        Self {
            flags: flags & ENCODE_FLAGS,
            encode_type: EncodeType::Encode,
            table: Rc::new(RefCell::new(EncodeTable::new(flags & ENCODE_FLAGS))),
            error: false,
        }
    }

    /// A mapper running the other way over the same table.
    ///
    /// SICADA-DIVERGE: upstream spells this as a copy constructor taking an
    /// `EncodeType`, which makes `EncodeMapper(mapper, DECODE)` and the plain
    /// copy constructor two different things that look alike. Here the reversal
    /// is named.
    pub fn inverse(&self) -> Self {
        Self {
            flags: self.flags,
            encode_type: match self.encode_type {
                EncodeType::Encode => EncodeType::Decode,
                EncodeType::Decode => EncodeType::Encode,
            },
            table: Rc::clone(&self.table),
            error: self.error,
        }
    }

    /// A mapper over an existing table.
    pub fn from_table(
        table: Rc<RefCell<EncodeTable<A::Label, A::Weight>>>,
        encode_type: EncodeType,
    ) -> Self {
        let flags = table.borrow().flags();
        Self {
            flags,
            encode_type,
            table,
            error: false,
        }
    }

    /// Which direction this runs in.
    pub fn encode_type(&self) -> EncodeType {
        self.encode_type
    }

    /// The encode flags.
    pub fn flags(&self) -> u8 {
        self.flags
    }

    /// The table, shared with every mapper made from this one.
    pub fn table(&self) -> &Rc<RefCell<EncodeTable<A::Label, A::Weight>>> {
        &self.table
    }

    /// Whether a decode has failed since this mapper was made.
    pub fn error(&self) -> bool {
        self.error
    }

    /// The input symbol table the encoded FST had.
    pub fn input_symbols(&self) -> Option<AtomicRc<SymbolTable>> {
        self.table.borrow().input_symbols()
    }

    /// The output symbol table the encoded FST had.
    pub fn output_symbols(&self) -> Option<AtomicRc<SymbolTable>> {
        self.table.borrow().output_symbols()
    }

    /// Remembers the input symbol table, so that decoding can put it back.
    pub fn set_input_symbols(&self, syms: Option<AtomicRc<SymbolTable>>) {
        self.table.borrow_mut().set_input_symbols(syms);
    }

    /// Remembers the output symbol table, so that decoding can put it back.
    pub fn set_output_symbols(&self, syms: Option<AtomicRc<SymbolTable>>) {
        self.table.borrow_mut().set_output_symbols(syms);
    }

    /// Whether `arc` is a final weight offered as an arc rather than a real
    /// transition.
    fn is_superfinal(arc: &A) -> bool {
        arc.nextstate() == A::StateId::no_state()
    }

    fn encode_arc(&mut self, arc: &A) -> A {
        // A hallucinated final transition is passed through untouched when
        // there is no weight to fold into a label, and when the state is not
        // final at all.
        if Self::is_superfinal(arc)
            && (self.flags & ENCODE_WEIGHTS == 0 || *arc.weight() == A::Weight::zero())
        {
            return arc.clone();
        }
        let label = self.table.borrow_mut().encode(arc);
        A::new(
            label,
            if self.flags & ENCODE_LABELS != 0 {
                label
            } else {
                arc.olabel()
            },
            if self.flags & ENCODE_WEIGHTS != 0 {
                A::Weight::one()
            } else {
                arc.weight().clone()
            },
            arc.nextstate(),
        )
    }

    fn decode_arc(&mut self, arc: &A) -> A {
        if Self::is_superfinal(arc) || arc.ilabel() == A::Label::epsilon() {
            return arc.clone();
        }
        if self.flags & ENCODE_LABELS != 0 && arc.ilabel() != arc.olabel() {
            self.error = true;
        }
        if self.flags & ENCODE_WEIGHTS != 0 && *arc.weight() != A::Weight::one() {
            self.error = true;
        }
        let table = self.table.borrow();
        let Some(triple) = table.decode(arc.ilabel()) else {
            self.error = true;
            return A::new(
                A::Label::no_label(),
                A::Label::no_label(),
                A::Weight::no_weight(),
                arc.nextstate(),
            );
        };
        if triple.ilabel == A::Label::no_label() {
            // The hallucinated triple a weighted superfinal transition was
            // given: it becomes an epsilon arc carrying the final weight, which
            // `rm_final_epsilon` then folds back into the state.
            return A::new(
                A::Label::epsilon(),
                A::Label::epsilon(),
                triple.weight.clone(),
                arc.nextstate(),
            );
        }
        A::new(
            triple.ilabel,
            if self.flags & ENCODE_LABELS != 0 {
                triple.olabel
            } else {
                arc.olabel()
            },
            if self.flags & ENCODE_WEIGHTS != 0 {
                triple.weight.clone()
            } else {
                arc.weight().clone()
            },
            arc.nextstate(),
        )
    }
}

impl<A: Arc> ArcMapper<A, A> for EncodeMapper<A>
where
    A::Weight: Hash + Eq,
{
    fn map(&mut self, arc: &A) -> A {
        match self.encode_type {
            EncodeType::Encode => self.encode_arc(arc),
            EncodeType::Decode => self.decode_arc(arc),
        }
    }

    fn final_action(&self) -> MapFinalAction {
        // Encoding a weight turns a final weight into an arc, so a superfinal
        // state is always needed; nothing else moves final weights.
        if self.encode_type == EncodeType::Encode && self.flags & ENCODE_WEIGHTS != 0 {
            MapFinalAction::RequireSuperfinal
        } else {
            MapFinalAction::NoSuperfinal
        }
    }

    fn input_symbols_action(&self) -> MapSymbolsAction {
        MapSymbolsAction::Clear
    }

    fn output_symbols_action(&self) -> MapSymbolsAction {
        MapSymbolsAction::Clear
    }

    fn properties(&self, inprops: u64) -> u64 {
        let mut outprops = inprops;
        if self.error {
            outprops |= K_ERROR;
        }
        let mut mask = K_FST_PROPERTIES;
        if self.flags & ENCODE_LABELS != 0 {
            mask &= K_I_LABEL_INVARIANT_PROPERTIES & K_O_LABEL_INVARIANT_PROPERTIES;
        }
        if self.flags & ENCODE_WEIGHTS != 0 {
            mask &= K_I_LABEL_INVARIANT_PROPERTIES
                & K_WEIGHT_INVARIANT_PROPERTIES
                & if self.encode_type == EncodeType::Encode {
                    K_ADD_SUPER_FINAL_PROPERTIES
                } else {
                    K_RM_SUPER_FINAL_PROPERTIES
                };
        }
        if self.encode_type == EncodeType::Encode {
            // Distinct triples get distinct labels, so no state can have two
            // arcs with the same input label going anywhere different.
            mask |= K_I_DETERMINISTIC;
        }
        outprops &= mask;
        if self.encode_type == EncodeType::Encode {
            if self.flags & ENCODE_LABELS != 0 {
                outprops |= K_ACCEPTOR;
            }
            if self.flags & ENCODE_WEIGHTS != 0 {
                outprops |= K_UNWEIGHTED | K_UNWEIGHTED_CYCLES;
            }
        }
        outprops
    }
}

/// Folds each arc's labels and weight into a single label, in place.
///
/// The FST's symbol tables are remembered in the mapper's table, and
/// [`decode`] puts them back.
///
/// Complexity: O(V + E).
pub fn encode<A, F>(fst: &mut F, mapper: &mut EncodeMapper<A>) -> Result<(), OpenFstError>
where
    A: Arc,
    A::Weight: Hash + Eq,
    F: MutableFst<A>,
{
    mapper.set_input_symbols(fst.input_symbols());
    mapper.set_output_symbols(fst.output_symbols());
    arc_map(fst, mapper)
}

/// Puts back the labels and weights [`encode`] folded away, in place.
///
/// SICADA-DIVERGE: upstream discovers a label with no entry in the table part
/// way through the rewrite, writes an arc carrying `NoWeight`, sets `kError`
/// and keeps going, so the FST comes back neither encoded nor decoded. Every
/// label is checked here before anything is rewritten, which costs one pass
/// over arcs the rewrite walks anyway, and the FST is left untouched if any of
/// them is bad.
pub fn decode<A, F>(fst: &mut F, mapper: &EncodeMapper<A>) -> Result<(), OpenFstError>
where
    A: Arc,
    A::Weight: Hash + Eq,
    F: MutableFst<A>,
{
    let mut decoder = mapper.inverse();
    debug_assert_eq!(decoder.encode_type(), EncodeType::Decode);
    check_decodable(fst, &decoder)?;

    arc_map(fst, &mut decoder)?;
    // Encoding a weight turned each final weight into an arc to a superfinal
    // state; decoding turned those back into epsilon arcs carrying the weight,
    // and this folds them back into final weights.
    rm_final_epsilon(fst);
    fst.set_input_symbols(mapper.input_symbols());
    fst.set_output_symbols(mapper.output_symbols());
    Ok(())
}

/// Reports the first arc `decoder` could not decode.
fn check_decodable<A, F>(fst: &F, decoder: &EncodeMapper<A>) -> Result<(), OpenFstError>
where
    A: Arc,
    A::Weight: Hash + Eq,
    F: Fst<A>,
{
    let flags = decoder.flags();
    let table = decoder.table.borrow();
    for state in fst.states() {
        for arc in fst.arcs(state) {
            if arc.ilabel() == A::Label::epsilon() {
                continue;
            }
            if flags & ENCODE_LABELS != 0 && arc.ilabel() != arc.olabel() {
                return Err(OpenFstError::InvalidOperation(format!(
                    "Decode: label-encoded arc from state {:?} has different input and output \
                     labels: {} and {}",
                    state,
                    arc.ilabel(),
                    arc.olabel()
                )));
            }
            if flags & ENCODE_WEIGHTS != 0 && *arc.weight() != A::Weight::one() {
                return Err(OpenFstError::InvalidOperation(format!(
                    "Decode: weight-encoded arc from state {:?} has non-trivial weight {}",
                    state,
                    arc.weight()
                )));
            }
            if table.decode(arc.ilabel()).is_none() {
                return Err(OpenFstError::InvalidOperation(format!(
                    "Decode: arc from state {:?} carries label {}, which the encode table does \
                     not have",
                    state,
                    arc.ilabel()
                )));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::test_support::{Rng, paths, random_acyclic_fst, string_weights};
    use crate::arc::StdArc;
    use crate::fst::ExpandedFst as _;
    use crate::fsts::vector_fst::StdVectorFst;
    use crate::properties::{K_ACCEPTOR, K_FST_PROPERTIES, K_UNWEIGHTED};
    use crate::weights::float_weight::TropicalWeight;

    fn mapper(flags: u8) -> EncodeMapper<StdArc> {
        EncodeMapper::new(flags)
    }

    /// 0 -a/1-> 1 -b/2-> 2, with 2 final at weight 3.
    fn chain() -> StdVectorFst {
        let mut fst = StdVectorFst::new();
        for _ in 0..3 {
            fst.add_state();
        }
        fst.set_start(0);
        fst.add_arc(0, StdArc::new(1, 10, TropicalWeight(1.0), 1));
        fst.add_arc(1, StdArc::new(2, 20, TropicalWeight(2.0), 2));
        fst.set_final(2, TropicalWeight(3.0));
        fst
    }

    fn observable(fst: &StdVectorFst) -> Vec<(Vec<i32>, Vec<i32>, String)> {
        string_weights(paths(fst, 12))
    }

    /// The contract of the whole file: whatever is folded away comes back.
    #[test]
    fn encoding_and_decoding_gives_back_the_same_paths() {
        let mut rng = Rng::new(0x_E4C0_DEED);
        for round in 0..200 {
            let fst = random_acyclic_fst(&mut rng, 6);
            let before = observable(&fst);
            for flags in [ENCODE_LABELS, ENCODE_WEIGHTS, ENCODE_FLAGS] {
                let mut encoder = mapper(flags);
                let mut copy = fst.clone();
                encode(&mut copy, &mut encoder).unwrap();
                decode(&mut copy, &encoder).unwrap();
                assert_eq!(observable(&copy), before, "round {round}, flags {flags}");
            }
        }
    }

    /// Encoding the labels makes the input and output sides agree, which is
    /// what determinization needs.
    #[test]
    fn encoding_labels_makes_an_acceptor() {
        let mut fst = chain();
        assert_eq!(fst.properties(K_ACCEPTOR, true) & K_ACCEPTOR, 0);

        let mut encoder = mapper(ENCODE_LABELS);
        encode(&mut fst, &mut encoder).unwrap();
        for state in 0..fst.num_states() as i32 {
            for arc in fst.arcs(state) {
                assert_eq!(arc.ilabel(), arc.olabel());
            }
        }
        assert_ne!(fst.properties(K_ACCEPTOR, true) & K_ACCEPTOR, 0);
        // The weights are untouched.
        assert_eq!(fst.final_weight(2), TropicalWeight(3.0));
    }

    /// Encoding the weights makes every arc weight One, as unweighted
    /// minimization requires.
    #[test]
    fn encoding_weights_makes_it_unweighted() {
        let mut fst = chain();
        let mut encoder = mapper(ENCODE_WEIGHTS);
        encode(&mut fst, &mut encoder).unwrap();
        for state in 0..fst.num_states() as i32 {
            for arc in fst.arcs(state) {
                assert_eq!(*arc.weight(), TropicalWeight::one());
            }
        }
        assert_ne!(fst.properties(K_UNWEIGHTED, true) & K_UNWEIGHTED, 0);
        // The output labels are untouched. The trailing epsilon is the arc
        // the final weight of state 2 became.
        let olabels: Vec<i32> = (0..fst.num_states() as i32)
            .flat_map(|s| fst.arcs(s).map(|a| a.olabel()).collect::<Vec<_>>())
            .collect();
        assert_eq!(olabels, vec![10, 20, 0]);
    }

    /// A final weight has nowhere to go once weights are labels, so it becomes
    /// an arc to a state added for the purpose, and decoding folds it back.
    #[test]
    fn a_final_weight_becomes_an_arc_and_comes_back() {
        let mut fst = chain();
        let before = fst.num_states();

        let mut encoder = mapper(ENCODE_WEIGHTS);
        encode(&mut fst, &mut encoder).unwrap();
        assert_eq!(fst.num_states(), before + 1, "a superfinal state was added");
        assert_eq!(
            fst.final_weight(2),
            TropicalWeight::zero(),
            "state 2 is no longer final; its weight left on an arc"
        );
        assert_eq!(fst.num_arcs(2), 1);

        decode(&mut fst, &encoder).unwrap();
        assert_eq!(fst.final_weight(2), TropicalWeight(3.0));
        assert_eq!(fst.num_arcs(2), 0);
    }

    /// An unweighted final state has nothing to fold away, so no arc is made
    /// for it and the state stays final.
    #[test]
    fn a_final_state_at_weight_one_keeps_its_final_weight() {
        let mut fst = chain();
        fst.set_final(2, TropicalWeight::one());
        let mut encoder = mapper(ENCODE_WEIGHTS);
        encode(&mut fst, &mut encoder).unwrap();
        // One triple for each of the two arcs, and one for the final weight.
        assert_eq!(encoder.table().borrow().len(), 3);
        decode(&mut fst, &encoder).unwrap();
        assert_eq!(fst.final_weight(2), TropicalWeight::one());
    }

    /// Two arcs alike in what is being encoded share a label; two that differ
    /// in it do not.
    #[test]
    fn arcs_share_a_label_exactly_when_what_is_encoded_matches() {
        let mut fst = StdVectorFst::new();
        for _ in 0..2 {
            fst.add_state();
        }
        fst.set_start(0);
        fst.add_arc(0, StdArc::new(1, 10, TropicalWeight(1.0), 1));
        fst.add_arc(0, StdArc::new(1, 10, TropicalWeight(1.0), 1)); // identical
        fst.add_arc(0, StdArc::new(1, 10, TropicalWeight(2.0), 1)); // other weight
        fst.add_arc(0, StdArc::new(1, 99, TropicalWeight(1.0), 1)); // other olabel

        let mut labels = fst.clone();
        let mut encoder = mapper(ENCODE_LABELS);
        encode(&mut labels, &mut encoder).unwrap();
        let got: Vec<i32> = labels.arcs(0).map(|a| a.ilabel()).collect();
        assert_eq!(
            got,
            vec![1, 1, 1, 2],
            "only the output label separates them when weights are not encoded"
        );

        let mut weights = fst.clone();
        let mut encoder = mapper(ENCODE_WEIGHTS);
        encode(&mut weights, &mut encoder).unwrap();
        let got: Vec<i32> = weights.arcs(0).map(|a| a.ilabel()).collect();
        assert_eq!(
            got,
            vec![1, 1, 2, 1],
            "only the weight separates them when labels are not encoded"
        );

        let mut both = fst;
        let mut encoder = mapper(ENCODE_FLAGS);
        encode(&mut both, &mut encoder).unwrap();
        let got: Vec<i32> = both.arcs(0).map(|a| a.ilabel()).collect();
        assert_eq!(got, vec![1, 1, 2, 3]);
    }

    /// Encoding clears the symbol tables, because the labels no longer mean
    /// what they meant; decoding puts them back.
    #[test]
    fn symbol_tables_are_kept_aside_and_put_back() {
        let mut syms = SymbolTable::new("input");
        syms.add_symbol("a", 1);
        let mut osyms = SymbolTable::new("output");
        osyms.add_symbol("A", 1);

        let mut fst = chain();
        fst.set_input_symbols(Some(AtomicRc::new(syms)));
        fst.set_output_symbols(Some(AtomicRc::new(osyms)));

        let mut encoder = mapper(ENCODE_FLAGS);
        encode(&mut fst, &mut encoder).unwrap();
        assert!(fst.input_symbols().is_none());
        assert!(fst.output_symbols().is_none());

        decode(&mut fst, &encoder).unwrap();
        assert_eq!(fst.input_symbols().unwrap().name(), "input");
        assert_eq!(fst.output_symbols().unwrap().name(), "output");
    }

    /// A label with no entry in the table is refused, and nothing is rewritten.
    #[test]
    fn decoding_a_label_the_table_does_not_have_is_refused() {
        let mut fst = chain();
        let mut encoder = mapper(ENCODE_FLAGS);
        encode(&mut fst, &mut encoder).unwrap();

        // A label one past the last one the table assigned.
        let past_the_end = encoder.table().borrow().len() as i32 + 1;
        fst.add_arc(
            0,
            StdArc::new(past_the_end, past_the_end, TropicalWeight::one(), 1),
        );
        let before: Vec<StdArc> = fst.arcs(0).collect();

        let err = decode(&mut fst, &encoder).unwrap_err();
        assert!(format!("{err}").contains("encode table"), "{err}");
        assert_eq!(
            fst.arcs(0).collect::<Vec<_>>(),
            before,
            "nothing was rewritten"
        );
    }

    /// An arc whose two sides disagree cannot have come from a label-encoded
    /// FST, so decoding refuses it rather than guessing.
    #[test]
    fn decoding_refuses_an_arc_that_was_never_encoded_that_way() {
        let mut fst = chain();
        let mut encoder = mapper(ENCODE_FLAGS);
        encode(&mut fst, &mut encoder).unwrap();
        fst.add_arc(0, StdArc::new(1, 2, TropicalWeight::one(), 1));
        assert!(decode(&mut fst, &encoder).is_err());

        let mut fst = chain();
        let mut encoder = mapper(ENCODE_WEIGHTS);
        encode(&mut fst, &mut encoder).unwrap();
        fst.add_arc(0, StdArc::new(1, 1, TropicalWeight(7.0), 1));
        assert!(decode(&mut fst, &encoder).is_err());
    }

    /// A decoder built before the encoder had finished still sees every label,
    /// which `encode → determinize → decode` relies on.
    #[test]
    fn a_decoder_shares_the_table_it_was_made_from() {
        let mut encoder = mapper(ENCODE_FLAGS);
        let decoder = encoder.inverse();
        assert_eq!(decoder.encode_type(), EncodeType::Decode);
        assert!(decoder.table().borrow().is_empty());

        let mut fst = chain();
        encode(&mut fst, &mut encoder).unwrap();
        assert_eq!(
            decoder.table().borrow().len(),
            encoder.table().borrow().len()
        );
        assert!(!decoder.table().borrow().is_empty());
    }

    /// The properties the mapper claims have to be the ones the result has.
    #[test]
    fn the_claimed_properties_are_the_ones_the_result_has() {
        let mut rng = Rng::new(0x_9E0_9E0);
        for round in 0..100 {
            let fst = random_acyclic_fst(&mut rng, 5);
            let inprops = fst.properties(K_FST_PROPERTIES, true);
            for flags in [ENCODE_LABELS, ENCODE_WEIGHTS, ENCODE_FLAGS] {
                let mut encoder = mapper(flags);
                let mut copy = fst.clone();
                let claimed = encoder.properties(inprops);
                encode(&mut copy, &mut encoder).unwrap();
                let actual = copy.properties(K_FST_PROPERTIES, true);
                assert_eq!(
                    claimed & !actual & K_FST_PROPERTIES,
                    0,
                    "round {round}, flags {flags}: claimed a property the result does not have: \
                     {:#x}",
                    claimed & !actual
                );
            }
        }
    }

    // --- The on-disk table format.

    /// The bytes OpenFst itself writes, taken from running its own code:
    /// `tests/oracles/encode-table-golden.cc`. Nothing about this layout
    /// may drift.
    #[test]
    fn the_table_matches_the_bytes_openfst_writes() {
        let mut table: EncodeTable<i32, TropicalWeight> = EncodeTable::new(ENCODE_FLAGS);
        table.encode(&StdArc::new(1, 2, TropicalWeight(0.5), 1));
        table.encode(&StdArc::new(3, 4, TropicalWeight(1.5), 1));

        let mut bytes = Vec::new();
        table.write(&mut bytes, ArcType::STANDARD).unwrap();

        #[rustfmt::skip]
        let golden: [u8; 49] = [
            0x4a, 0x6d, 0xd9, 0x7e,                          // magic 2128178506
            0x08, 0x00, 0x00, 0x00,                          // arc type length 8
            b's', b't', b'a', b'n', b'd', b'a', b'r', b'd',  // "standard"
            0x03,                                            // flags: labels|weights
            0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // two triples
            0x01, 0x00, 0x00, 0x00,                          // ilabel 1
            0x02, 0x00, 0x00, 0x00,                          // olabel 2
            0x00, 0x00, 0x00, 0x3f,                          // weight 0.5
            0x03, 0x00, 0x00, 0x00,                          // ilabel 3
            0x04, 0x00, 0x00, 0x00,                          // olabel 4
            0x00, 0x00, 0xc0, 0x3f,                          // weight 1.5
        ];
        assert_eq!(bytes, golden);
    }

    #[test]
    fn a_table_round_trips_through_bytes() {
        let mut table: EncodeTable<i32, TropicalWeight> = EncodeTable::new(ENCODE_FLAGS);
        for (ilabel, olabel, weight) in [(1, 2, 0.5), (3, 4, 1.5), (1, 2, 2.5)] {
            table.encode(&StdArc::new(ilabel, olabel, TropicalWeight(weight), 1));
        }
        let mut syms = SymbolTable::new("input");
        syms.add_symbol("a", 1);
        table.set_input_symbols(Some(AtomicRc::new(syms)));

        let mut bytes = Vec::new();
        table.write(&mut bytes, ArcType::STANDARD).unwrap();
        let read: EncodeTable<i32, TropicalWeight> =
            EncodeTable::read(&mut bytes.as_slice()).unwrap();

        assert_eq!(read.len(), table.len());
        assert_eq!(read.flags(), table.flags());
        for label in 1..=table.len() as i32 {
            assert_eq!(read.decode(label), table.decode(label), "label {label}");
        }
        assert_eq!(read.input_symbols().unwrap().name(), "input");
        assert!(read.output_symbols().is_none());
    }

    /// The pre-2019 format had no arc type and a 32-bit flags field. It is
    /// still readable.
    #[test]
    fn the_deprecated_header_is_still_readable() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&ENCODE_DEPRECATED_MAGIC_NUMBER.to_le_bytes());
        bytes.extend_from_slice(&(ENCODE_FLAGS as u32).to_le_bytes());
        bytes.extend_from_slice(&1i64.to_le_bytes());
        bytes.extend_from_slice(&7i32.to_le_bytes());
        bytes.extend_from_slice(&8i32.to_le_bytes());
        bytes.extend_from_slice(&0.25f32.to_le_bytes());

        let table: EncodeTable<i32, TropicalWeight> =
            EncodeTable::read(&mut bytes.as_slice()).unwrap();
        assert_eq!(table.len(), 1);
        assert_eq!(
            table.decode(1),
            Some(&Triple {
                ilabel: 7,
                olabel: 8,
                weight: TropicalWeight(0.25)
            })
        );
    }

    #[test]
    fn a_stream_that_is_not_an_encode_table_is_refused() {
        let bytes = 12345i32.to_le_bytes();
        assert!(EncodeTable::<i32, TropicalWeight>::read(&mut bytes.as_slice()).is_err());
    }

    /// A count larger than the triples that follow fails on the short read
    /// rather than reserving for it.
    #[test]
    fn a_table_claiming_more_triples_than_it_has_is_refused() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&ENCODE_MAGIC_NUMBER.to_le_bytes());
        bytes.extend_from_slice(&8i32.to_le_bytes());
        bytes.extend_from_slice(b"standard");
        bytes.push(ENCODE_FLAGS);
        bytes.extend_from_slice(&u64::MAX.to_le_bytes());
        assert!(EncodeTable::<i32, TropicalWeight>::read(&mut bytes.as_slice()).is_err());
    }
}
