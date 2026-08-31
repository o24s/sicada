//! Forced alignment against a reference, solved exactly.
//!
//! Alignment is decoding with the answer already known. The reference is a flat
//! sequence of `N` phones, the acoustic model has scored `T` frames, and the
//! only question left is which frames each phone occupies. The graph is
//! therefore a single chain rather than a lattice of everything that could have
//! been said, and a search over one chain is small enough to do exactly.
//!
//! # The chain
//!
//! States `s_0 … s_N`, where `s_i` means "the first `i` phones are behind us".
//! `s_0` is the start, `s_N` is the only final state, and four transitions
//! leave each state, every one of them consuming exactly one frame:
//!
//! | transition | reads | goes to | costs | means |
//! |---|---|---|---|---|
//! | hold blank | blank | `s_i` | 0 | silence, or phone `i + 1` has not started |
//! | hold phone | phone `i` | `s_i` | 0 | phone `i` is still sounding (`i ≥ 1`) |
//! | commit | phone `i + 1` | `s_{i+1}` | 0 | phone `i + 1` starts in this frame |
//! | skip | blank | `s_{i+1}` | `skip(i + 1)` | phone `i + 1` never happens |
//!
//! The skip transition makes the reference advisory rather than binding: a line
//! of the script that went unspoken, or a verse the singer dropped, costs `skip`
//! per phone instead of forcing the rest of the alignment to absorb it.
//! [`AlignChain::new`] forbids skipping, and a caller opts in per phone, because
//! the cost is a property of the phone rather than of the aligner. See
//! [`AlignChain::with_skip_costs`].
//!
//! # Why the emission matrix is not widened
//!
//! Two transitions out of `s_i` read the blank column and two read a phone
//! column, so the chain is *non-deterministic on its input labels*: from one
//! state, two different arcs carry the same label. A decoder that indexes the
//! emissions by arc label rather than by column cannot express that, and has to
//! be handed a matrix widened to one column per arc, `T × (C + 2N)` instead of
//! `T × C`. For a ten-minute utterance against a 5 385-phone reference that is
//! 1.22 GiB, and 2.44 GiB once it is copied. Every added column is a duplicate:
//! a skip channel *is* the blank column, and position `i`'s repeat channel *is*
//! `phone(i)`'s.
//!
//! Nothing here pays that cost. [`align`] reads columns out of
//! [`DenseFst::frame`](crate::dense::DenseFst::frame) directly and never forms
//! a label at all, and [`AlignChain::to_fst`], which does form labels, is still
//! read column-wise by [`viterbi_decode`](crate::viterbi::viterbi_decode).
//! sicada's decoders ask the matrix which column an arc names, rather than
//! requiring the matrix to be reshaped to suit the arcs.
//!
//! # Why there is no beam
//!
//! Because every transition consumes one frame and advances at most one phone,
//! a path of `T` frames stands at position `i` in frame `t` only when
//! `t - (T - N) ≤ i ≤ t`, and [`band`](crate::trellis::band) fills exactly
//! those cells. That is not a beam but the set of cells that lie on a complete
//! path at all, so the search is exhaustive and there is nothing to tune.
//!
//! It is also the cheaper of the two. The four transitions into a cell fit in
//! two bits, so the whole traceback for the ten-minute case is
//! `30 241 × 5 386 / 4 = 41 MB`, alongside two rows of scores at 43 KB. A
//! *pruned* search over the same chain keeps a token per surviving state per
//! frame, or `T × max_active × 24 B`, which at k2's default of 30 000 active
//! states is 21.8 GiB, which does not fit on a 30 GiB machine.
//!
//! Being exact removes a failure mode as well as the memory. A beam narrow
//! enough to finish quickly has to be paid for with a smaller skip cost, and a
//! smaller skip cost *improves the acoustic score*, since every phone given up
//! is a frame explained by the blank instead. On the ten-minute case a beam of
//! 25 with the skip cost dropped to 0.75 reached the end, scored better than the
//! right answer (0.273 against 0.354 nats per frame), and lost the times of
//! 19.7 % of the reference without reporting anything. Nothing inside the
//! decoder can distinguish the two cases, so the fix is not to have the knob.
//!
//! # When the chain is not the shape you want
//!
//! A real alignment usually wants something this chain does not have: a whole
//! word given up at once rather than a phone at a time, a phone that has to
//! last two frames, a penalty that leans on a voice-activity detector frame by
//! frame. None of that needs a new solver.
//!
//! Everything above the phones lives in [`trellis`](crate::trellis), which takes
//! the transitions entering a cell (how many there are, what each costs, what it
//! means) and supplies the band, the packed traceback and the forward-backward.
//! [`AlignChain`] is one implementation of that trait and [`ChainTrellis`] is
//! where to read it; [`align`] is [`best_path`] over
//! it plus the reading-back that turns codes into phones. A caller with their
//! own topology writes the trait and keeps the rest, and a caller who wants this
//! chain's raw path calls [`AlignChain::against`] and
//! [`best_path`] themselves.

use std::ops::Range;

use sicada::arc::{Arc, ArcLabel, ArcStateId};
use sicada::error::OpenFstError;
use sicada::fst::{Fst, MutableFst};
use sicada::fsts::vector_fst::VectorFst;
use sicada::properties::K_FST_PROPERTIES;
use sicada::weight::Weight;

use crate::dense::{DenseFst, FromScore};
use crate::trellis::{Path, ReversibleTrellis, Step, Trellis, best_path};

/// Whether each of the chain's four transitions sounds the phone of the
/// position it lands in, indexed by code.
const SOUNDS: [bool; 4] = [false, true, true, false];

/// A reference to align: the phones in order, and what each one costs to give
/// up.
///
/// Phones are *columns* of the acoustic matrix, not labels. Which label a
/// column sits on matters only to [`to_fst`](Self::to_fst), because only an FST
/// has labels; [`align`] never forms one.
#[derive(Debug, Clone, PartialEq)]
pub struct AlignChain {
    /// `phones[p]` is the column position `p` sounds.
    phones: Vec<u32>,
    /// `skips[p]` is the cost of giving position `p` up. Infinite forbids it.
    skips: Vec<f32>,
    blank: u32,
}

impl AlignChain {
    /// A reference every phone of which has to be given frames.
    ///
    /// Skipping is forbidden until a caller asks for it, because a skip cost is
    /// a claim about the phone; see [`with_skip_costs`](Self::with_skip_costs).
    /// Column 0 is the blank, which is where a CTC model puts it;
    /// [`with_blank`](Self::with_blank) moves it.
    pub fn new(phones: impl Into<Vec<u32>>) -> Self {
        let phones = phones.into();
        Self {
            skips: vec![f32::INFINITY; phones.len()],
            phones,
            blank: 0,
        }
    }

    /// The cost of giving up each position, one per phone.
    ///
    /// An infinite cost forbids the skip, and [`new`](Self::new) leaves one
    /// everywhere. A finite cost is a threshold: the phone is dropped
    /// only when keeping it costs strictly more, so the number that matters is
    /// how much acoustic evidence the phone is worth.
    ///
    /// The cost belongs to the phone rather than to the utterance. In the
    /// reference measurements a Japanese `cl`, the closure of a geminate, is set
    /// to 1.0 against 6.0 for everything else, because it *is* silence and so no
    /// positive evidence for it can exist even in principle. Over 80 read
    /// utterances that fires zero times and moves no boundary in the third
    /// decimal, while over four sung ones it fires 113 times.
    ///
    /// Widening it to the pause phones is the tempting next step, and it is a
    /// mistake: in one song it cost 35 of 37 punctuation marks their times,
    /// against 1 with `cl` alone, for no measurable gain.
    ///
    /// # Errors
    ///
    /// A count that is not the number of phones, or a cost below zero or not a
    /// number. A negative cost would pay the alignment to throw the reference
    /// away.
    pub fn with_skip_costs(mut self, costs: &[f32]) -> Result<Self, OpenFstError> {
        if costs.len() != self.phones.len() {
            return Err(OpenFstError::InvalidOperation(format!(
                "AlignChain: {} skip costs for {} phones",
                costs.len(),
                self.phones.len()
            )));
        }
        if let Some(bad) = costs.iter().position(|cost| cost.is_nan() || *cost < 0.0) {
            return Err(OpenFstError::InvalidOperation(format!(
                "AlignChain: the skip cost at position {bad} is {}, and a skip that pays for \
                 itself would drop the reference rather than align it",
                costs[bad]
            )));
        }
        self.skips.copy_from_slice(costs);
        Ok(self)
    }

    /// The same cost for every position.
    ///
    /// # Errors
    ///
    /// A cost below zero or not a number, as [`with_skip_costs`](Self::with_skip_costs).
    pub fn with_uniform_skip_cost(self, cost: f32) -> Result<Self, OpenFstError> {
        let costs = vec![cost; self.phones.len()];
        self.with_skip_costs(&costs)
    }

    /// The column meaning "no phone is sounding". Column 0 by default.
    pub fn with_blank(mut self, column: u32) -> Self {
        self.blank = column;
        self
    }

    /// The number of phones in the reference.
    #[inline(always)]
    pub fn num_phones(&self) -> usize {
        self.phones.len()
    }

    /// Whether the reference is empty, in which case every frame is blank.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.phones.is_empty()
    }

    /// The columns the reference sounds, in order.
    #[inline(always)]
    pub fn phones(&self) -> &[u32] {
        &self.phones
    }

    /// The cost of giving up each position, in order.
    #[inline(always)]
    pub fn skip_costs(&self) -> &[f32] {
        &self.skips
    }

    /// The column that means silence.
    #[inline(always)]
    pub fn blank(&self) -> u32 {
        self.blank
    }

    /// The chain against a matrix of scores, ready to be solved.
    ///
    /// [`align`] is this plus [`best_path`] plus reading the answer back into
    /// phones. Call it directly to get at the [`Path`], whose `codes` are the
    /// four below and whose `positions` are reference positions, or to run
    /// [`posteriors`](crate::trellis::posteriors) with an accumulator of your
    /// own.
    ///
    /// # Errors
    ///
    /// A phone, or the blank, naming a column the acoustic model does not have.
    /// Checked once here so that solving never has to.
    pub fn against<'a, A>(
        &'a self,
        dense: &'a DenseFst<'a, A>,
    ) -> Result<ChainTrellis<'a, A>, OpenFstError>
    where
        A: Arc,
        A::Weight: FromScore,
    {
        self.check_columns(dense.num_symbols())?;
        Ok(ChainTrellis { chain: self, dense })
    }

    /// The transition that stays put and says nothing: silence, or a phone that
    /// has not started.
    pub const HOLD_BLANK: u8 = 0;
    /// The transition that stays put sounding the phone it is on.
    pub const HOLD_PHONE: u8 = 1;
    /// The transition that moves to the next phone and sounds it.
    pub const COMMIT: u8 = 2;
    /// The transition that moves to the next phone without sounding it.
    pub const SKIP: u8 = 3;

    /// Whether a transition sounds the phone of the position it lands in.
    ///
    /// The four codes are listed best-first, which is the tie-break; see
    /// [`ChainTrellis::steps_into`](crate::trellis::Trellis::steps_into).
    ///
    /// # Panics
    ///
    /// On a code this chain has no transition for.
    #[inline(always)]
    pub const fn sounds(code: u8) -> bool {
        SOUNDS[code as usize]
    }

    /// The column a frame reads when it sounds `position`, or the blank when it
    /// sounds nothing.
    #[inline(always)]
    fn column(&self, position: Option<usize>) -> u32 {
        match position {
            Some(p) => self.phones[p],
            None => self.blank,
        }
    }

    /// Reports a column the acoustic model does not have.
    ///
    /// Checked once, so the inner loop can index the frame unconditionally.
    pub(crate) fn check_columns(&self, num_symbols: usize) -> Result<(), OpenFstError> {
        let named = std::iter::once((None, self.blank)).chain(
            self.phones
                .iter()
                .enumerate()
                .map(|(p, &column)| (Some(p), column)),
        );
        for (position, column) in named {
            if column as usize >= num_symbols {
                let what = match position {
                    Some(p) => format!("position {p}"),
                    None => "the blank".to_string(),
                };
                return Err(OpenFstError::InvalidOperation(format!(
                    "AlignChain: {what} is column {column}, which a {num_symbols}-symbol acoustic \
                     matrix does not have"
                )));
            }
        }
        Ok(())
    }

    /// The chain as an ordinary FST, so it can go through the decoders in this
    /// crate or be composed like anything else.
    ///
    /// `label_offset` has to be the one the
    /// [`DenseFst`] was built with. It is 1 by default,
    /// because label 0 is epsilon to every FST algorithm and a blank is not one.
    /// Input labels are the columns, offset.
    ///
    /// Output labels name what the frame sounded: `p + 1` for position `p`, and
    /// `N + 1` for a frame that sounded nothing. *Every* arc carries one, so a
    /// decoded path's output labels are one per frame and are the alignment
    /// itself, which [`Alignment::from_output_labels`] reads back. That is what
    /// makes [`lattice_decode`](crate::lattice::lattice_decode) and
    /// [`n_best`](crate::nbest::n_best) over this FST produce *alternative*
    /// alignments, which the exact aligner, returning one answer, does not.
    ///
    /// [`align`] is what to use to align. This exists to put the chain in front
    /// of the rest of the library, and it is also the oracle the exact aligner
    /// is tested against, since the two share no code.
    ///
    /// # Errors
    ///
    /// A label that does not fit the arc's label type, or an offset below 1,
    /// which would put a column on epsilon and so on an arc consuming no frame.
    pub fn to_fst<A: Arc>(&self, label_offset: i64) -> Result<VectorFst<A>, OpenFstError>
    where
        A::Weight: FromScore,
    {
        if label_offset < 1 {
            return Err(OpenFstError::InvalidOperation(
                "AlignChain::to_fst: column 0 would be epsilon, which consumes no frame".into(),
            ));
        }
        let fits = |value: i64, what: &str| -> Result<A::Label, OpenFstError> {
            A::Label::from_i64(value).ok_or_else(|| {
                OpenFstError::InvalidOperation(format!(
                    "AlignChain::to_fst: {what} {value} does not fit the arc's label type"
                ))
            })
        };
        let input = |column: u32| fits(label_offset + column as i64, "input label");
        let n = self.phones.len();
        // `sounds(Some(p))` is p + 1 and `sounds(None)` is N + 1, so the two
        // never collide and neither is epsilon.
        let sounds = |position: Option<usize>| {
            let value = match position {
                Some(p) => p as i64 + 1,
                None => n as i64 + 1,
            };
            fits(value, "output label")
        };

        let mut fst: VectorFst<A> = VectorFst::new();
        fst.reserve_states(n + 1);
        for _ in 0..=n {
            fst.add_state();
        }
        fst.set_start(A::StateId::from_usize(0));
        fst.set_final(A::StateId::from_usize(n), A::Weight::one());

        let blank = input(self.blank)?;
        let silent = sounds(None)?;
        for i in 0..=n {
            let from = A::StateId::from_usize(i);
            let to = A::StateId::from_usize((i + 1).min(n));

            fst.add_arc(from, A::new(blank, silent, A::Weight::one(), from));
            if i > 0 {
                let held = input(self.phones[i - 1])?;
                fst.add_arc(
                    from,
                    A::new(held, sounds(Some(i - 1))?, A::Weight::one(), from),
                );
            }
            if i < n {
                let next = input(self.phones[i])?;
                fst.add_arc(from, A::new(next, sounds(Some(i))?, A::Weight::one(), to));
                // An infinite cost is the absence of the arc, not an arc of
                // weight zero: `Weight::zero()` would still be an arc, and
                // algorithms are entitled to keep it.
                let cost = self.skips[i];
                if cost.is_finite() {
                    fst.add_arc(from, A::new(blank, silent, A::Weight::from_cost(cost), to));
                }
            }
        }

        fst.properties(K_FST_PROPERTIES, true);
        Ok(fst)
    }
}

/// Which phone each frame sounded.
///
/// Frames the reference does not account for sound nothing, and belong to no
/// phone; see [`spans`](Self::spans) for why that convention and not the other.
#[derive(Debug, Clone, PartialEq)]
pub struct Alignment {
    /// Per frame, one more than the position sounding in it, or 0 for none.
    ///
    /// Held as one number rather than a position and a bit so that a frame
    /// costs four bytes, and so that no caller can pair a position with the
    /// wrong bit.
    sounding: Vec<u32>,
    num_phones: usize,
    cost: f32,
}

impl Alignment {
    /// The number of frames aligned.
    #[inline(always)]
    pub fn num_frames(&self) -> usize {
        self.sounding.len()
    }

    /// The number of phones in the reference this came from.
    #[inline(always)]
    pub fn num_phones(&self) -> usize {
        self.num_phones
    }

    /// The position sounding in `frame`, or `None` for a frame that sounded
    /// nothing.
    ///
    /// # Panics
    ///
    /// If `frame` is past the last one.
    #[inline(always)]
    pub fn sounding(&self, frame: usize) -> Option<usize> {
        (self.sounding[frame] as usize).checked_sub(1)
    }

    /// The whole alignment, one entry per frame.
    pub fn frames(&self) -> impl ExactSizeIterator<Item = Option<usize>> + '_ {
        self.sounding.iter().map(|&k| (k as usize).checked_sub(1))
    }

    /// The alignment's total cost: the acoustic scores of every frame, plus the
    /// cost of each phone given up.
    #[inline(always)]
    pub fn cost(&self) -> f32 {
        self.cost
    }

    /// The frames each position occupies, as `[first frame sounding it, last
    /// frame sounding it + 1)`, or `None` for a position no frame sounded.
    ///
    /// **Frames left blank belong to no phone.** The other convention, that a
    /// phone owns everything up to the next one, measures the same to within
    /// the accuracy of phone boundaries themselves (against the labels shipped
    /// with the PJS corpus, n = 754, both put the median onset error at 49 ms
    /// and the offset error at 53 against 57 ms), but it breaks at the end of a
    /// line. Where silence or an interlude follows, the line's last phone
    /// swallows the whole gap, measured at 2.1 s: in silence the model scores
    /// the blank and a pause alike, so Viterbi has no reason to commit the pause
    /// early.
    pub fn spans(&self) -> Vec<Option<Range<usize>>> {
        let mut spans = vec![None; self.num_phones];
        for (frame, &sounding) in self.sounding.iter().enumerate() {
            let Some(position) = (sounding as usize).checked_sub(1) else {
                continue;
            };
            match &mut spans[position] {
                slot @ None => *slot = Some(frame..frame + 1),
                Some(span) => span.end = frame + 1,
            }
        }
        spans
    }

    /// The frames each *group* of consecutive positions occupies, given how
    /// many positions each group holds.
    ///
    /// The reference is flat, so a word is a run of positions, and this is how
    /// word times come out of a phone alignment: pass the phone count of each
    /// word. A group takes its first sounding frame to its last, ignoring the
    /// blanks in between, since a word does not stop existing because it has a
    /// pause in the middle of it.
    ///
    /// A group every phone of which was given up has no span at all. That case is
    /// worth handling rather than papering over: a line with no time is a line
    /// that was not sung.
    ///
    /// # Errors
    ///
    /// Sizes that do not add up to the number of phones aligned.
    pub fn group_spans(&self, sizes: &[usize]) -> Result<Vec<Option<Range<usize>>>, OpenFstError> {
        let total: usize = sizes.iter().sum();
        if total != self.num_phones {
            return Err(OpenFstError::InvalidOperation(format!(
                "Alignment: groups of {total} phones for a {}-phone reference",
                self.num_phones
            )));
        }
        let spans = self.spans();
        let mut grouped = Vec::with_capacity(sizes.len());
        let mut at = 0;
        for &size in sizes {
            let mut group: Option<Range<usize>> = None;
            for span in spans[at..at + size].iter().flatten() {
                group = Some(match group {
                    None => span.clone(),
                    Some(so_far) => so_far.start..span.end,
                });
            }
            grouped.push(group);
            at += size;
        }
        Ok(grouped)
    }

    /// The positions no frame sounded: the phones the alignment gave up.
    ///
    /// Their share of the reference is the diagnostic that matters. On the
    /// reference material a correct alignment skips about 1 %; a skip cost
    /// low enough to let a narrow beam reach the end skipped 19.7 % while
    /// scoring *better* acoustically, which is why [`align`] has no beam.
    pub fn skipped(&self) -> Vec<usize> {
        let mut sounded = vec![false; self.num_phones];
        for &sounding in &self.sounding {
            if let Some(position) = (sounding as usize).checked_sub(1) {
                sounded[position] = true;
            }
        }
        sounded
            .into_iter()
            .enumerate()
            .filter_map(|(position, sounded)| (!sounded).then_some(position))
            .collect()
    }

    /// What each frame paid the acoustic model, in order.
    ///
    /// Recomputed from the matrix rather than remembered: a frame that sounded
    /// nothing paid the blank column, one that sounded position `p` paid `p`'s.
    /// [`mean_acoustic_cost`](Self::mean_acoustic_cost) is available precisely
    /// because the alignment can say this without searching again.
    ///
    /// # Panics
    ///
    /// If `chain` is not the one this was aligned against, or `dense` not the
    /// matrix it was aligned to.
    pub fn acoustic_costs<'a, A>(
        &'a self,
        chain: &'a AlignChain,
        dense: &'a DenseFst<'a, A>,
    ) -> impl ExactSizeIterator<Item = f32> + 'a
    where
        A: Arc + 'a,
        A::Weight: FromScore,
    {
        self.sounding.iter().enumerate().map(move |(frame, &k)| {
            let column = chain.column((k as usize).checked_sub(1));
            dense.frame(frame)[column as usize]
        })
    }

    /// The mean of [`acoustic_costs`](Self::acoustic_costs), or `0.0` for no
    /// frames.
    ///
    /// This is the one automatic warning that the reference is not what was
    /// said. Costs here are negative log probabilities, so smaller is better: a
    /// correct transcript measured 0.11 to 0.35 nats per frame, and an
    /// unrelated one 1.78. It is *not* usable for choosing the skip cost, or any
    /// other search setting, since giving up more of the reference always
    /// improves it.
    pub fn mean_acoustic_cost<A>(&self, chain: &AlignChain, dense: &DenseFst<'_, A>) -> f32
    where
        A: Arc,
        A::Weight: FromScore,
    {
        if self.sounding.is_empty() {
            return 0.0;
        }
        let total: f64 = self
            .sounding
            .iter()
            .enumerate()
            .map(|(frame, &k)| {
                let column = chain.column((k as usize).checked_sub(1));
                dense.frame(frame)[column as usize] as f64
            })
            .sum();
        (total / self.sounding.len() as f64) as f32
    }

    /// Reads an alignment back from a [`Path`] through
    /// [`AlignChain::against`].
    ///
    /// [`align`] is this on the path [`best_path`] returns. It is separate
    /// because the path is the more general answer: a caller may want the
    /// transitions themselves, or may have solved the chain alongside a
    /// topology of their own.
    ///
    /// # Errors
    ///
    /// A code the chain has no transition for, which means the path came from
    /// a different trellis.
    pub fn from_path(chain: &AlignChain, path: &Path) -> Result<Self, OpenFstError> {
        let mut sounding = Vec::with_capacity(path.num_frames());
        for (frame, (&code, &position)) in path.codes().iter().zip(path.positions()).enumerate() {
            let sounds = *SOUNDS.get(code as usize).ok_or_else(|| {
                OpenFstError::InvalidOperation(format!(
                    "Alignment: transition {code} at frame {frame} is not one of the chain's four"
                ))
            })?;
            sounding.push(if sounds { position } else { 0 });
        }
        Ok(Self {
            sounding,
            num_phones: chain.phones.len(),
            cost: path.cost(),
        })
    }

    /// Reads an alignment back from a path through
    /// [`AlignChain::to_fst`](AlignChain::to_fst).
    ///
    /// The chain's arcs all carry an output label, so a decoded path's labels
    /// are one per frame. That turns
    /// [`lattice_decode`](crate::lattice::lattice_decode) and
    /// [`n_best`](crate::nbest::n_best) over the chain into alternative
    /// alignments; [`align`] returns the best one directly.
    ///
    /// # Errors
    ///
    /// A label naming no position, which means the path did not come from this
    /// chain.
    pub fn from_output_labels<L: ArcLabel>(
        chain: &AlignChain,
        labels: &[L],
        cost: f32,
    ) -> Result<Self, OpenFstError> {
        let num_phones = chain.phones.len();
        let silent = num_phones as i64 + 1;
        let mut sounding = Vec::with_capacity(labels.len());
        for (frame, label) in labels.iter().enumerate() {
            let value = label.to_i64().unwrap_or(-1);
            if value == silent {
                sounding.push(0);
            } else if value >= 1 && value < silent {
                sounding.push(value as u32);
            } else {
                return Err(OpenFstError::InvalidOperation(format!(
                    "Alignment: output label {value} at frame {frame} names no position of a \
                     {num_phones}-phone reference"
                )));
            }
        }
        Ok(Self {
            sounding,
            num_phones,
            cost,
        })
    }
}

/// [`AlignChain`] against a matrix of scores: the trellis [`align`] solves.
///
/// The chain alone has no scores, and a [`Trellis`] is the two together. This
/// is public because the trellis is the reusable half: `best_path` and
/// `posteriors` take one, so a caller wanting the raw [`Path`], or a variant
/// topology of their own, starts here rather than at [`align`]. It is also the
/// worked example the [`trellis`](crate::trellis) docs point at.
#[derive(Debug, Clone, Copy)]
pub struct ChainTrellis<'a, A: Arc> {
    chain: &'a AlignChain,
    dense: &'a DenseFst<'a, A>,
}

impl<A: Arc> ChainTrellis<'_, A> {
    /// The reference this reads.
    #[inline(always)]
    pub fn chain(&self) -> &AlignChain {
        self.chain
    }
}

impl<A: Arc> Trellis<4> for ChainTrellis<'_, A>
where
    A::Weight: FromScore,
{
    type Frame<'f>
        = &'f [f32]
    where
        Self: 'f;

    #[inline(always)]
    fn num_frames(&self) -> usize {
        self.dense.num_frames()
    }

    #[inline(always)]
    fn num_positions(&self) -> usize {
        self.chain.phones.len()
    }

    #[inline(always)]
    fn frame(&self, frame: usize) -> &[f32] {
        self.dense.frame(frame)
    }

    /// The four, in the order that is the tie-break: waiting beats sounding,
    /// standing still beats advancing, and keeping a phone beats giving it up.
    /// Only a strict improvement moves off the incumbent, which makes a skip
    /// cost a threshold rather than a suggestion.
    #[inline(always)]
    fn steps_into(&self, frame: &[f32], position: usize) -> [Step; 4] {
        let blank = Step::new(0, frame[self.chain.blank as usize]);
        if position == 0 {
            return [blank, Step::ABSENT, Step::ABSENT, Step::ABSENT];
        }
        let phone = frame[self.chain.phones[position - 1] as usize];
        [
            blank,
            Step::new(0, phone),
            Step::new(1, phone),
            Step::new(1, self.chain.skips[position - 1] + blank.cost),
        ]
    }
}

impl<A: Arc> ReversibleTrellis<4> for ChainTrellis<'_, A>
where
    A::Weight: FromScore,
{
    /// Written out rather than left to
    /// [`derive_steps_out_of`](crate::trellis::derive_steps_out_of), which asks
    /// what enters each cell within reach and reads the answer off that.
    /// Measured at 5.32 s against 5.63 s for a forward-backward over ten minutes
    /// of audio, so the 6 % is worth having. The price is that the two readings
    /// can now disagree, and
    /// [`axioms::check`](crate::trellis::axioms::check) is what settles that.
    #[inline(always)]
    fn steps_out_of(&self, frame: &[f32], position: usize) -> [Step; 4] {
        let blank = Step::new(0, frame[self.chain.blank as usize]);
        let hold = if position > 0 {
            Step::new(0, frame[self.chain.phones[position - 1] as usize])
        } else {
            Step::ABSENT
        };
        let (commit, skip) = if position < self.chain.phones.len() {
            (
                Step::new(1, frame[self.chain.phones[position] as usize]),
                Step::new(1, self.chain.skips[position] + blank.cost),
            )
        } else {
            (Step::ABSENT, Step::ABSENT)
        };
        [blank, hold, commit, skip]
    }
}

/// The column a transition reads, given the cell it lands in.
///
/// A frame either sounds the phone of the position it ends at or says nothing,
/// so this needs the target rather than the source.
#[inline(always)]
pub(crate) fn column_read(chain: &AlignChain, code: u8, position: usize) -> u32 {
    if SOUNDS[code as usize] {
        chain.phones[position - 1]
    } else {
        chain.blank
    }
}

/// Aligns `chain` to `dense`: the best path of the chain against the acoustic
/// scores, exactly.
///
/// This is [`best_path`] over [`AlignChain::against`], read back into phones.
/// A caller who wants the path itself, either to interpret the transitions their
/// own way or because they have replaced the chain with a topology of their own,
/// should call those two directly; see [`trellis`](crate::trellis).
///
/// Returns `None` when no path exists, which means a reference longer than the
/// audio with no skips to make up the difference.
///
/// # Errors
///
/// A phone naming a column the acoustic model does not have, which is a
/// mismatch between the reference and the model rather than a bad alignment; or
/// a matrix so large that the traceback plane does not fit in memory, reported
/// rather than attempted.
pub fn align<A>(
    chain: &AlignChain,
    dense: &DenseFst<'_, A>,
) -> Result<Option<Alignment>, OpenFstError>
where
    A: Arc,
    A::Weight: FromScore,
{
    let trellis = chain.against(dense)?;
    let Some(path) = best_path(&trellis)? else {
        return Ok(None);
    };
    Alignment::from_path(chain, &path).map(Some)
}

#[cfg(test)]
mod tests {
    use super::*;
    use sicada::arc::StdArc;
    use sicada::fst::ExpandedFst;
    use sicada::fsts::vector_fst::StdVectorFst;

    use crate::compact::{DeterminizeLatticeOptions, determinize_lattice};
    use crate::frontier::DecodeOptions;
    use crate::lattice::{LatticeDecodeOptions, lattice_decode};
    use crate::nbest::n_best;
    use crate::trellis::axioms;
    use crate::viterbi::viterbi_decode;

    /// Blank plus three phones.
    const SYMBOLS: usize = 4;

    /// Scores that make one column nearly certain in each frame.
    fn certain(columns: &[usize]) -> Vec<f32> {
        let mut scores = vec![10.0; columns.len() * SYMBOLS];
        for (frame, &column) in columns.iter().enumerate() {
            scores[frame * SYMBOLS + column] = 0.0;
        }
        scores
    }

    /// What the alignment says it cost, recomputed from the reference and the
    /// matrix: the frames' acoustic scores plus the phones given up.
    ///
    /// A traceback that has drifted off the winning path still reports the
    /// winning *cost*, so comparing against an oracle's cost alone would not
    /// catch it. This does.
    fn recomputed_cost(
        alignment: &Alignment,
        chain: &AlignChain,
        dense: &DenseFst<'_, StdArc>,
    ) -> f32 {
        let acoustic: f32 = alignment.acoustic_costs(chain, dense).sum();
        let skipped: f32 = alignment
            .skipped()
            .into_iter()
            .map(|position| chain.skip_costs()[position])
            .sum();
        acoustic + skipped
    }

    /// The answer the aligner is supposed to agree with: build the same chain as
    /// an ordinary FST and decode it with the general decoder.
    ///
    /// The two share no code: one walks a hash-map frontier over an FST's arcs,
    /// the other a banded array of `f32`. An agreement between them is therefore
    /// evidence about the recurrence rather than about a shared mistake.
    fn by_decoding(chain: &AlignChain, dense: &DenseFst<'_, StdArc>) -> Option<Alignment> {
        let fst: StdVectorFst = chain.to_fst(1).expect("a chain FST");
        let decoded =
            viterbi_decode(&fst, dense, &DecodeOptions::exhaustive()).expect("a decode")?;
        Some(
            Alignment::from_output_labels(chain, &decoded.labels, decoded.weight.0)
                .expect("labels from this chain"),
        )
    }

    #[test]
    fn a_phone_owns_the_frames_that_sound_it() {
        // Phone 1 for two frames, then silence, then phone 2.
        let scores = certain(&[1, 1, 0, 2]);
        let dense = DenseFst::<StdArc>::new(&scores, 4, SYMBOLS).unwrap();
        let chain = AlignChain::new(vec![1, 2]);

        let alignment = align(&chain, &dense).unwrap().expect("an alignment");
        assert_eq!(
            alignment.frames().collect::<Vec<_>>(),
            vec![Some(0), Some(0), None, Some(1)]
        );
        assert_eq!(alignment.spans(), vec![Some(0..2), Some(3..4)]);
        assert!(alignment.skipped().is_empty());
        assert!(alignment.cost().abs() < 1e-6, "{}", alignment.cost());
    }

    /// Word times out of a phone alignment, which is usually what a caller
    /// wants from one.
    #[test]
    fn a_group_of_phones_spans_its_first_sounding_frame_to_its_last() {
        // Two words of two phones. The second word's second phone has no
        // evidence anywhere and is given up.
        let scores = certain(&[1, 0, 2, 0, 3, 0]);
        let dense = DenseFst::<StdArc>::new(&scores, 6, SYMBOLS).unwrap();
        let chain = AlignChain::new(vec![1, 2, 3, 2])
            .with_uniform_skip_cost(1.0)
            .unwrap();
        let alignment = align(&chain, &dense).unwrap().expect("an alignment");

        assert_eq!(alignment.skipped(), vec![3]);
        // The first word runs across the silence between its two phones.
        assert_eq!(
            alignment.group_spans(&[2, 2]).unwrap(),
            vec![Some(0..3), Some(4..5)]
        );
        // A word every phone of which was given up has no time at all.
        assert_eq!(
            alignment.group_spans(&[3, 1]).unwrap(),
            vec![Some(0..5), None]
        );
        assert_eq!(alignment.group_spans(&[4]).unwrap(), vec![Some(0..5)]);

        let err = alignment.group_spans(&[2, 1]).unwrap_err();
        assert!(format!("{err}").contains("groups of 3 phones"), "{err}");
    }

    /// The convention the whole crate's timings rest on: a blank frame is
    /// nobody's.
    #[test]
    fn a_blank_frame_belongs_to_no_phone() {
        // One phone, then eight frames of silence: the end of a line.
        let scores = certain(&[1, 0, 0, 0, 0, 0, 0, 0, 0]);
        let dense = DenseFst::<StdArc>::new(&scores, 9, SYMBOLS).unwrap();
        let chain = AlignChain::new(vec![1]);

        let alignment = align(&chain, &dense).unwrap().expect("an alignment");
        assert_eq!(
            alignment.spans(),
            vec![Some(0..1)],
            "the phone must not swallow the silence after it"
        );
    }

    #[test]
    fn an_empty_reference_leaves_every_frame_sounding_nothing() {
        let scores = certain(&[1, 2, 0]);
        let dense = DenseFst::<StdArc>::new(&scores, 3, SYMBOLS).unwrap();
        let chain = AlignChain::new(vec![]);

        let alignment = align(&chain, &dense).unwrap().expect("an alignment");
        assert!(alignment.frames().all(|sounding| sounding.is_none()));
        assert_eq!(alignment.spans(), vec![]);
        // Three frames of blank, two of which the model dislikes.
        assert!(
            (alignment.cost() - 20.0).abs() < 1e-6,
            "{}",
            alignment.cost()
        );
    }

    #[test]
    fn a_reference_longer_than_the_audio_aligns_to_nothing() {
        let scores = certain(&[1, 2]);
        let dense = DenseFst::<StdArc>::new(&scores, 2, SYMBOLS).unwrap();
        let chain = AlignChain::new(vec![1, 2, 3]);
        assert_eq!(align(&chain, &dense).unwrap(), None);

        // Not even with skips: a skip consumes a frame like everything else.
        let chain = chain.with_uniform_skip_cost(0.0).unwrap();
        assert_eq!(align(&chain, &dense).unwrap(), None);
    }

    #[test]
    fn a_phone_the_model_has_no_column_for_is_reported() {
        let scores = certain(&[1]);
        let dense = DenseFst::<StdArc>::new(&scores, 1, SYMBOLS).unwrap();

        let err = align(&AlignChain::new(vec![9]), &dense).unwrap_err();
        assert!(format!("{err}").contains("position 0 is column 9"), "{err}");

        let err = align(&AlignChain::new(vec![1]).with_blank(7), &dense).unwrap_err();
        assert!(format!("{err}").contains("the blank is column 7"), "{err}");
    }

    /// The reason skipping exists: text that was never spoken.
    ///
    /// Note what a skip is actually weighed against. It consumes a frame like
    /// every other transition, and that frame reads the blank, so giving up a
    /// phone is worth it when `skip` is less than what sounding the phone costs
    /// *over* falling silent, rather than less than what sounding it costs.
    #[test]
    fn a_phone_with_no_evidence_is_given_up_only_when_that_is_cheaper() {
        // Two frames sure of phone 1, then two the model hears as silence. The
        // third frame is where phone 2 fits best, and even there it costs 3
        // more than the blank.
        let scores = [
            10.0, 0.0, 10.0, 10.0, //
            10.0, 0.0, 10.0, 10.0, //
            0.0, 10.0, 3.0, 10.0, //
            0.0, 10.0, 10.0, 10.0,
        ];
        let dense = DenseFst::<StdArc>::new(&scores, 4, SYMBOLS).unwrap();
        let reference = vec![1, 2];

        // Under that 3, the phone goes.
        let cheap = AlignChain::new(reference.clone())
            .with_skip_costs(&[6.0, 1.0])
            .unwrap();
        let alignment = align(&cheap, &dense).unwrap().expect("an alignment");
        assert_eq!(alignment.skipped(), vec![1]);
        assert_eq!(alignment.spans()[0], Some(0..2));
        assert_eq!(alignment.spans()[1], None);
        assert!(
            (alignment.cost() - 1.0).abs() < 1e-6,
            "{}",
            alignment.cost()
        );

        // Over it, the phone comes back, in the frame that fits it best.
        let dear = AlignChain::new(reference)
            .with_skip_costs(&[6.0, 5.0])
            .unwrap();
        let alignment = align(&dear, &dense).unwrap().expect("an alignment");
        assert!(alignment.skipped().is_empty());
        assert_eq!(alignment.spans(), vec![Some(0..2), Some(2..3)]);
        assert!(
            (alignment.cost() - 3.0).abs() < 1e-6,
            "{}",
            alignment.cost()
        );
    }

    /// The threshold has to be strict, or a skip cost set to exactly the
    /// evidence against the phone would throw it away.
    #[test]
    fn a_skip_that_only_ties_does_not_happen() {
        // Frame 1 hears silence; sounding phone 2 there costs 4 more.
        let scores = [
            10.0, 0.0, 10.0, 10.0, //
            0.0, 10.0, 4.0, 10.0,
        ];
        let dense = DenseFst::<StdArc>::new(&scores, 2, SYMBOLS).unwrap();
        let reference = vec![1, 2];

        let tied = AlignChain::new(reference.clone())
            .with_skip_costs(&[9.0, 4.0])
            .unwrap();
        let alignment = align(&tied, &dense).unwrap().expect("an alignment");
        assert!(
            alignment.skipped().is_empty(),
            "a tie has to keep the reference"
        );
        assert_eq!(alignment.spans(), vec![Some(0..1), Some(1..2)]);
        assert!(
            (alignment.cost() - 4.0).abs() < 1e-6,
            "{}",
            alignment.cost()
        );

        // A hair under, and it is a skip: the threshold is where it says.
        let under = AlignChain::new(reference)
            .with_skip_costs(&[9.0, 3.9])
            .unwrap();
        let alignment = align(&under, &dense).unwrap().expect("an alignment");
        assert_eq!(alignment.skipped(), vec![1]);
    }

    #[test]
    fn a_skip_cost_that_pays_for_itself_is_refused() {
        let chain = AlignChain::new(vec![1, 2]);
        let err = chain.clone().with_skip_costs(&[1.0, -1.0]).unwrap_err();
        assert!(format!("{err}").contains("position 1"), "{err}");
        assert!(chain.clone().with_skip_costs(&[f32::NAN, 1.0]).is_err());
        assert!(
            chain.clone().with_skip_costs(&[1.0]).is_err(),
            "wrong count"
        );
        assert!(chain.with_uniform_skip_cost(-0.5).is_err());
    }

    #[test]
    fn the_alignment_recovers_what_each_frame_paid() {
        let scores = certain(&[1, 0, 2]);
        let dense = DenseFst::<StdArc>::new(&scores, 3, SYMBOLS).unwrap();
        let chain = AlignChain::new(vec![1, 2]);
        let alignment = align(&chain, &dense).unwrap().expect("an alignment");

        assert_eq!(
            alignment.acoustic_costs(&chain, &dense).collect::<Vec<_>>(),
            vec![0.0, 0.0, 0.0]
        );
        assert_eq!(alignment.mean_acoustic_cost(&chain, &dense), 0.0);

        // A reference the audio does not contain costs every frame instead.
        let wrong = AlignChain::new(vec![3, 3]);
        let alignment = align(&wrong, &dense).unwrap().expect("an alignment");
        assert!(
            alignment.mean_acoustic_cost(&wrong, &dense) > 5.0,
            "an unrelated reference has to be visible in the per-frame cost"
        );
    }

    /// A small xorshift, so the random cases below are the same every run.
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }

        fn below(&mut self, n: usize) -> usize {
            (self.next() % n as u64) as usize
        }

        /// A cost on a fine enough grid that two paths rarely tie, so the two
        /// searches' tie-breaking rarely has to agree for the alignments to.
        fn cost(&mut self) -> f32 {
            self.below(1 << 20) as f32 / 4096.0
        }
    }

    /// Every alignment of `num_frames` frames onto `chain`, scored directly.
    ///
    /// Exponential, so only for the smallest cases, but it shares nothing at all
    /// with the aligner, not even the shape of the recurrence.
    fn by_brute_force(
        chain: &AlignChain,
        dense: &DenseFst<'_, StdArc>,
        num_frames: usize,
    ) -> Option<f32> {
        fn walk(
            chain: &AlignChain,
            dense: &DenseFst<'_, StdArc>,
            num_frames: usize,
            frame: usize,
            position: usize,
            cost: f32,
            best: &mut Option<f32>,
        ) {
            if frame == num_frames {
                if position == chain.num_phones() && best.is_none_or(|so_far| cost < so_far) {
                    *best = Some(cost);
                }
                return;
            }
            let scores = dense.frame(frame);
            let blank = scores[chain.blank() as usize];
            let mut step = |position, extra: f32| {
                walk(
                    chain,
                    dense,
                    num_frames,
                    frame + 1,
                    position,
                    cost + extra,
                    best,
                )
            };
            step(position, blank);
            if position > 0 {
                step(position, scores[chain.phones()[position - 1] as usize]);
            }
            if position < chain.num_phones() {
                step(position + 1, scores[chain.phones()[position] as usize]);
                let skip = chain.skip_costs()[position];
                if skip.is_finite() {
                    step(position + 1, skip + blank);
                }
            }
        }

        let mut best = None;
        walk(chain, dense, num_frames, 0, 0, 0.0, &mut best);
        best
    }

    /// Against every alignment there is, on cases small enough to enumerate.
    #[test]
    fn it_agrees_with_enumerating_every_alignment() {
        let mut rng = Rng(0x1234_5678_9ABC_DEF1);
        let mut compared = 0;

        for round in 0..200 {
            let num_frames = 1 + rng.below(7);
            let num_phones = rng.below(4);
            let phones: Vec<u32> = (0..num_phones)
                .map(|_| 1 + rng.below(SYMBOLS - 1) as u32)
                .collect();
            let chain = AlignChain::new(phones);
            // Half the rounds allow skipping, at a cost worth about one frame.
            let chain = if rng.below(2) == 0 {
                chain
                    .with_uniform_skip_cost(rng.below(1 << 12) as f32 / 512.0)
                    .unwrap()
            } else {
                chain
            };

            let scores: Vec<f32> = (0..num_frames * SYMBOLS).map(|_| rng.cost()).collect();
            let dense = DenseFst::<StdArc>::new(&scores, num_frames, SYMBOLS).unwrap();

            let expected = by_brute_force(&chain, &dense, num_frames);
            let alignment = align(&chain, &dense).unwrap();

            match (expected, alignment) {
                (None, None) => {}
                (Some(expected), Some(alignment)) => {
                    compared += 1;
                    assert!(
                        (alignment.cost() - expected).abs() < 1e-3,
                        "round {round}: aligner {} against every path's best {expected}",
                        alignment.cost()
                    );
                    // And the path it reports is the path it priced.
                    assert!(
                        (recomputed_cost(&alignment, &chain, &dense) - alignment.cost()).abs()
                            < 1e-3,
                        "round {round}: the traceback does not add up to the cost"
                    );
                    assert_eq!(alignment.num_frames(), num_frames);
                }
                (expected, alignment) => {
                    panic!("round {round}: brute force {expected:?}, aligner {alignment:?}")
                }
            }
        }

        assert!(compared > 150, "only {compared} rounds had an alignment");
    }

    /// Against the same chain decoded as an ordinary FST, at sizes brute force
    /// cannot reach, which is where the band and the packed traceback start to
    /// matter.
    #[test]
    fn it_agrees_with_decoding_the_chain_as_an_fst() {
        let mut rng = Rng(0xFEED_FACE_1234_5678);
        let mut compared = 0;

        for round in 0..200 {
            let num_frames = 1 + rng.below(40);
            let num_phones = rng.below(12);
            let phones: Vec<u32> = (0..num_phones)
                .map(|_| 1 + rng.below(SYMBOLS - 1) as u32)
                .collect();
            let chain = AlignChain::new(phones);
            let chain = if rng.below(2) == 0 {
                chain
                    .with_uniform_skip_cost(rng.below(1 << 12) as f32 / 512.0)
                    .unwrap()
            } else {
                chain
            };

            let scores: Vec<f32> = (0..num_frames * SYMBOLS).map(|_| rng.cost()).collect();
            let dense = DenseFst::<StdArc>::new(&scores, num_frames, SYMBOLS).unwrap();

            let expected = by_decoding(&chain, &dense);
            let alignment = align(&chain, &dense).unwrap();

            match (expected, alignment) {
                (None, None) => {}
                (Some(expected), Some(alignment)) => {
                    compared += 1;
                    assert!(
                        (alignment.cost() - expected.cost()).abs() < 1e-2,
                        "round {round}: aligner {} against the decoder {}",
                        alignment.cost(),
                        expected.cost()
                    );
                    assert!(
                        (recomputed_cost(&alignment, &chain, &dense) - alignment.cost()).abs()
                            < 1e-2,
                        "round {round}: the traceback does not add up to the cost"
                    );
                    assert_eq!(expected.num_frames(), num_frames, "one label per frame");
                }
                (expected, alignment) => {
                    panic!("round {round}: decoder {expected:?}, aligner {alignment:?}")
                }
            }
        }

        assert!(compared > 150, "only {compared} rounds had an alignment");
    }

    /// The chain is an FST like any other, so the lattice decoder gives
    /// alternative alignments the exact aligner does not.
    #[test]
    fn the_chain_decodes_to_alternative_alignments() {
        // Two frames sure of phone 1, and one in between that is torn between
        // holding it and falling silent, so the alignment is either three frames
        // of phone 1 or two with a gap.
        let scores = [
            9.0, 0.0, 9.0, 9.0, //
            1.0, 0.0, 9.0, 9.0, //
            9.0, 0.0, 9.0, 9.0,
        ];
        let dense = DenseFst::<StdArc>::new(&scores, 3, SYMBOLS).unwrap();
        let chain = AlignChain::new(vec![1]);
        let fst: StdVectorFst = chain.to_fst(1).unwrap();

        let lattice = lattice_decode(&fst, &dense, &LatticeDecodeOptions::exhaustive())
            .unwrap()
            .expect("a lattice");
        let compact = determinize_lattice(&lattice, &DeterminizeLatticeOptions::default()).unwrap();
        let answers = n_best(&compact, 2).unwrap();
        assert_eq!(answers.len(), 2);

        let best = Alignment::from_output_labels(&chain, &answers[0].words, answers[0].cost())
            .expect("an alignment");
        assert_eq!(best.spans(), vec![Some(0..3)], "the phone held throughout");
        assert_eq!(
            align(&chain, &dense).unwrap().unwrap().spans(),
            best.spans(),
            "and it is what the exact aligner returns"
        );

        let second = Alignment::from_output_labels(&chain, &answers[1].words, answers[1].cost())
            .expect("an alignment");
        assert_eq!(
            second.frames().collect::<Vec<_>>(),
            vec![Some(0), None, Some(0)]
        );
        assert!((second.cost() - best.cost() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn labels_from_another_chain_are_reported() {
        let chain = AlignChain::new(vec![1, 2]);
        // 3 is the silent label for a 2-phone reference; 4 names nothing.
        assert!(Alignment::from_output_labels(&chain, &[1i32, 3], 0.0).is_ok());
        let err = Alignment::from_output_labels(&chain, &[1i32, 4], 0.0).unwrap_err();
        assert!(format!("{err}").contains("names no position"), "{err}");
        assert!(Alignment::from_output_labels(&chain, &[0i32], 0.0).is_err());
    }

    #[test]
    fn a_chain_fst_puts_its_columns_where_the_matrix_has_them() {
        let chain = AlignChain::new(vec![1, 2])
            .with_uniform_skip_cost(1.0)
            .unwrap();
        let fst: StdVectorFst = chain.to_fst(1).unwrap();
        assert_eq!(fst.num_states(), 3);
        // s_0: hold blank, commit, skip. s_1: those plus hold phone. s_2: hold
        // blank and hold phone.
        assert_eq!(fst.num_arcs(0), 3);
        assert_eq!(fst.num_arcs(1), 4);
        assert_eq!(fst.num_arcs(2), 2);
        assert!(
            fst.states()
                .all(|s| fst.arcs(s).all(|arc| arc.ilabel() != 0)),
            "every arc has to consume a frame"
        );

        // Forbidding the skips removes the arcs rather than zero-weighting them.
        let fst: StdVectorFst = AlignChain::new(vec![1, 2]).to_fst(1).unwrap();
        assert_eq!(fst.num_arcs(0), 2);
        assert!(AlignChain::new(vec![1]).to_fst::<StdArc>(0).is_err());
    }

    /// The contract the solvers rely on, run as the checker every trellis is
    /// told to run, including the requirement that the chain's hand-written
    /// backward reading is the one its forward reading implies.
    #[test]
    fn the_chain_obeys_the_trellis_contract() {
        let chain = AlignChain::new(vec![1, 2, 1])
            .with_skip_costs(&[1.0, 2.0, f32::INFINITY])
            .unwrap();
        let scores: Vec<f32> = (0..4 * SYMBOLS).map(|i| i as f32 / 3.0).collect();
        let dense = DenseFst::<StdArc>::new(&scores, 4, SYMBOLS).unwrap();
        axioms::check(&chain.against(&dense).unwrap());

        // And with skipping forbidden everywhere, which is a different set of
        // absent transitions.
        let rigid = AlignChain::new(vec![1, 2, 1]);
        axioms::check(&rigid.against(&dense).unwrap());
        axioms::check(&AlignChain::new(vec![]).against(&dense).unwrap());
    }

    /// The band is an exact reachability argument, so its edges have to be
    /// right at both ends: a reference exactly as long as the audio leaves no
    /// slack at all.
    #[test]
    fn a_reference_as_long_as_the_audio_has_one_alignment() {
        let scores = certain(&[1, 2, 3]);
        let dense = DenseFst::<StdArc>::new(&scores, 3, SYMBOLS).unwrap();
        let chain = AlignChain::new(vec![1, 2, 3]);

        let alignment = align(&chain, &dense).unwrap().expect("an alignment");
        assert_eq!(
            alignment.frames().collect::<Vec<_>>(),
            vec![Some(0), Some(1), Some(2)]
        );
        assert!(alignment.cost().abs() < 1e-6);

        // Even when every frame would rather be blank.
        let scores = [0.0, 10.0, 10.0, 10.0].repeat(3);
        let dense = DenseFst::<StdArc>::new(&scores, 3, SYMBOLS).unwrap();
        let alignment = align(&chain, &dense).unwrap().expect("an alignment");
        assert_eq!(
            alignment.frames().collect::<Vec<_>>(),
            vec![Some(0), Some(1), Some(2)]
        );
    }
}
