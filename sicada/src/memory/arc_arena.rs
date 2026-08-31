//! Allocator for contiguous runs of arcs.
//!
//! Port of OpenFst's `arc-arena.h`. An [`ArcArena`] hands out contiguous arc
//! arrays without a separate heap allocation per state:
//!
//! ```text
//! for each state:
//!     for each arc:
//!         arena.push_arc(arc);
//!     let run = arena.commit_arcs();   // or arena.drop_arcs() to reuse the space
//! ```
//!
//! A committed run stays valid until [`ArcArena::clear`] is called, even as later
//! runs are appended and the arena grows.
//!
//! Unlike the C++ original, which returns a raw `const Arc *` that the caller
//! must keep alive by convention, [`commit_arcs`](ArcArena::commit_arcs) returns
//! an [`ArcRun`] handle resolved through [`ArcArena::arcs`]. That keeps the arena
//! non-self-referential: the borrow checker enforces the "runs die at `clear()`"
//! rule that upstream can only document, and no `unsafe` is needed to store the
//! handles alongside the arena.

/// Number of arcs in the arena's first block, when unspecified.
const DEFAULT_BLOCK_SIZE: usize = 256;

/// Upper bound on the block size carried across [`ArcArena::clear`], when
/// unspecified. Caps the memory an intermittent burst leaves behind.
const DEFAULT_MAX_RETAINED_SIZE: usize = 1_000_000;

/// Handle to a committed run of arcs inside an [`ArcArena`].
///
/// Valid until the arena is cleared. Resolve it with [`ArcArena::arcs`].
/// SICADA-OPT: three `u32`s rather than three `usize`s. A handle is stored per
/// state, so halving it halves the memory a caller walks when it revisits its
/// runs; no arena block can hold four billion arcs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ArcRun {
    block: u32,
    start: u32,
    len: u32,
}

impl ArcRun {
    /// Number of arcs in the run.
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.len as usize
    }

    /// Whether the run is empty.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// Arena allocating contiguous runs of arcs out of geometrically growing blocks.
///
/// Generic over the element type rather than over `A: Arc`; nothing here depends
/// on the arc interface, and composition-style algorithms reuse it for other
/// per-state payloads.
pub struct ArcArena<A> {
    /// Blocks that are no longer written to, kept alive because committed runs
    /// point into them.
    retired: Vec<Vec<A>>,
    /// The block being filled.
    ///
    /// SICADA-OPT: held apart from `retired` so that `push_arc` touches one
    /// vector directly. Reaching it through `blocks.last_mut()` costs a bounds
    /// check and an `Option` on the hottest path in state expansion, on top of
    /// the capacity check the push itself performs.
    current: Vec<A>,
    /// Offset in `current` where the in-progress run starts.
    start: usize,
    /// Size of a freshly grown block, unless a larger one is requested.
    block_size: usize,
    /// Size of the block retained across `clear`.
    first_block_size: usize,
    /// Total capacity across all live blocks.
    total_size: usize,
    /// Cap on `first_block_size` when it is recomputed by `clear`.
    max_retained_size: usize,
}

impl<A> Default for ArcArena<A> {
    fn default() -> Self {
        Self::new()
    }
}

impl<A> ArcArena<A> {
    /// Creates an arena with the default block size and retention cap.
    pub fn new() -> Self {
        Self::with_options(DEFAULT_BLOCK_SIZE, DEFAULT_MAX_RETAINED_SIZE)
    }

    /// Creates an arena whose blocks start at `block_size` arcs.
    pub fn with_block_size(block_size: usize) -> Self {
        Self::with_options(block_size, DEFAULT_MAX_RETAINED_SIZE)
    }

    /// Creates an arena with an explicit block size and retention cap.
    ///
    /// `max_retained_size` bounds how much space [`clear`](Self::clear) keeps for
    /// the next round, so one unusually large batch does not pin its peak
    /// footprint forever.
    pub fn with_options(block_size: usize, max_retained_size: usize) -> Self {
        let block_size = block_size.max(1);
        Self {
            retired: Vec::new(),
            current: Vec::with_capacity(block_size),
            start: 0,
            block_size,
            first_block_size: block_size,
            total_size: block_size,
            max_retained_size,
        }
    }

    /// Total capacity, in arcs, across all live blocks.
    ///
    /// Mirrors upstream's `Size()`: allocated capacity, not the number of arcs
    /// pushed.
    #[inline(always)]
    pub fn size(&self) -> usize {
        self.total_size
    }

    /// The run currently being built, not yet committed.
    #[inline(always)]
    pub fn pending(&self) -> &[A] {
        &self.current[self.start..]
    }

    /// Resolves a run returned by [`commit_arcs`](Self::commit_arcs).
    ///
    /// # Panics
    ///
    /// Panics if the run came from a different arena, or from this one before a
    /// [`clear`](Self::clear).
    #[inline(always)]
    pub fn arcs(&self, run: ArcRun) -> &[A] {
        let block = run.block as usize;
        let block = if block == self.retired.len() {
            &self.current
        } else {
            &self.retired[block]
        };
        let start = run.start as usize;
        &block[start..start + run.len as usize]
    }

    /// Ensures `n` more arcs fit in the current block without a further growth.
    pub fn reserve_arcs(&mut self, n: usize) {
        if self.current.capacity() - self.current.len() >= n {
            return;
        }
        self.new_block(n);
    }

    /// Appends one arc to the in-progress run.
    ///
    /// SICADA-OPT: the capacity test has to happen here anyway, since a block
    /// must never reallocate or committed runs would stop being contiguous, and
    /// `Vec::push` would then repeat it. Writing through the pointer skips the
    /// second check, leaving one comparison per arc, as upstream's
    /// `next_ == end_` does.
    #[inline]
    pub fn push_arc(&mut self, arc: A) {
        let len = self.current.len();
        if len == self.current.capacity() {
            // Upstream grows to twice the in-progress run's length; `new_block`
            // takes the requested *additional* capacity, hence the pending length.
            self.new_block((len - self.start).max(1));
            self.current.push(arc);
            return;
        }
        // SAFETY: `len` is below the capacity, so the slot is allocated and
        // uninitialized, and the length is corrected immediately after the write.
        unsafe {
            std::ptr::write(self.current.as_mut_ptr().add(len), arc);
            self.current.set_len(len + 1);
        }
    }

    /// Commits the in-progress run and returns a handle to it.
    ///
    /// Upstream calls this `GetArcs()`.
    pub fn commit_arcs(&mut self) -> ArcRun {
        let block = self.retired.len();
        let start = self.start;
        let len = self.current.len() - start;
        self.start += len;
        debug_assert!(block <= u32::MAX as usize && start + len <= u32::MAX as usize);
        ArcRun {
            block: block as u32,
            start: start as u32,
            len: len as u32,
        }
    }

    /// Discards the in-progress run, returning its space to the arena.
    pub fn drop_arcs(&mut self) {
        self.current.truncate(self.start);
    }

    /// Releases every block and invalidates all outstanding [`ArcRun`]s.
    ///
    /// The arena restarts with a single block large enough to hold everything the
    /// previous round used, capped at `max_retained_size`, so repeated rounds stop
    /// paying for growth.
    pub fn clear(&mut self) {
        self.retired.clear();
        if self.total_size > self.first_block_size {
            self.first_block_size = self.max_retained_size.min(self.total_size);
            self.current = Vec::with_capacity(self.first_block_size);
        } else {
            self.current.clear();
        }
        self.total_size = self.first_block_size;
        self.start = 0;
    }

    /// Allocates a new block with room for the in-progress run plus `n` more arcs,
    /// and moves the in-progress run into it.
    fn new_block(&mut self, n: usize) {
        let pending_len = self.current.len() - self.start;
        // SICADA-BUGFIX: upstream sizes the new block `max(n, block_size_)` and
        // then copies the `length` in-progress arcs into it, overflowing the
        // block whenever `length` exceeds that. The run being moved has to fit.
        let new_block_size = (pending_len + n).max(self.block_size);
        let mut new_block = Vec::with_capacity(new_block_size);
        new_block.extend(self.current.drain(self.start..));
        self.total_size += new_block_size;
        self.retired
            .push(std::mem::replace(&mut self.current, new_block));
        self.start = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arc::{Arc, StdArc};
    use crate::weight::Weight;
    use crate::weights::float_weight::TropicalWeight;
    use std::cell::Cell;
    use std::rc::Rc;

    fn arc(label: i32) -> StdArc {
        StdArc::new(label, label, TropicalWeight::one(), label)
    }

    fn push_run(arena: &mut ArcArena<StdArc>, labels: std::ops::Range<i32>) -> ArcRun {
        for label in labels {
            arena.push_arc(arc(label));
        }
        arena.commit_arcs()
    }

    #[test]
    fn committed_runs_are_contiguous_and_in_order() {
        let mut arena = ArcArena::with_block_size(4);
        let run = push_run(&mut arena, 0..3);

        let arcs = arena.arcs(run);
        assert_eq!(arcs.len(), 3);
        assert_eq!(
            arcs.iter().map(|a| a.ilabel).collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
    }

    #[test]
    fn earlier_runs_stay_valid_after_the_arena_grows() {
        // The whole point of the arena: a handed-out run must survive any number
        // of later allocations, including the block growth they trigger.
        let mut arena = ArcArena::with_block_size(4);
        let runs: Vec<_> = (0..8)
            .map(|i| push_run(&mut arena, i * 3..i * 3 + 3))
            .collect();

        for (i, run) in runs.iter().enumerate() {
            let expected: Vec<i32> = (i as i32 * 3..i as i32 * 3 + 3).collect();
            assert_eq!(
                arena
                    .arcs(*run)
                    .iter()
                    .map(|a| a.ilabel)
                    .collect::<Vec<_>>(),
                expected,
                "run {i} was invalidated"
            );
        }
    }

    #[test]
    fn a_run_longer_than_a_block_stays_contiguous() {
        let mut arena = ArcArena::with_block_size(4);
        let run = push_run(&mut arena, 0..37);

        let arcs = arena.arcs(run);
        assert_eq!(arcs.len(), 37);
        assert!(arcs.iter().enumerate().all(|(i, a)| a.ilabel == i as i32));
    }

    #[test]
    fn reserve_arcs_mid_sequence_keeps_the_run_contiguous() {
        // Regression test for the upstream heap overflow: reserving space while a
        // run longer than the requested reservation is already in progress.
        let mut arena = ArcArena::with_block_size(4);
        for label in 0..300 {
            arena.push_arc(arc(label));
        }
        arena.reserve_arcs(250);
        assert_eq!(arena.pending().len(), 300);
        for label in 300..400 {
            arena.push_arc(arc(label));
        }

        let run = arena.commit_arcs();
        let arcs = arena.arcs(run);
        assert_eq!(arcs.len(), 400);
        assert!(arcs.iter().enumerate().all(|(i, a)| a.ilabel == i as i32));
    }

    #[test]
    fn reserve_arcs_does_not_grow_when_the_block_already_fits() {
        let mut arena = ArcArena::<StdArc>::with_block_size(64);
        let before = arena.size();
        arena.reserve_arcs(64);
        assert_eq!(arena.size(), before);
    }

    #[test]
    fn drop_arcs_reuses_the_space() {
        let mut arena = ArcArena::with_block_size(16);
        let first = push_run(&mut arena, 0..4);

        for label in 100..104 {
            arena.push_arc(arc(label));
        }
        arena.drop_arcs();
        assert!(arena.pending().is_empty());

        let second = push_run(&mut arena, 200..202);
        assert_eq!(arena.arcs(first).len(), 4);
        assert_eq!(
            arena
                .arcs(second)
                .iter()
                .map(|a| a.ilabel)
                .collect::<Vec<_>>(),
            vec![200, 201]
        );
        // The dropped arcs were overwritten rather than leaked into a new block.
        assert_eq!(arena.size(), 16);
    }

    #[test]
    fn clear_retains_capacity_for_the_next_round() {
        let mut arena = ArcArena::with_block_size(4);
        push_run(&mut arena, 0..40);
        let grown = arena.size();
        assert!(grown > 4);

        arena.clear();
        assert_eq!(arena.size(), grown, "clear should retain the peak capacity");

        // The retained block absorbs the same workload without growing again.
        push_run(&mut arena, 0..40);
        assert_eq!(arena.size(), grown);
    }

    #[test]
    fn clear_caps_retained_capacity() {
        let mut arena = ArcArena::with_options(4, 16);
        push_run(&mut arena, 0..100);
        arena.clear();
        assert_eq!(arena.size(), 16);
    }

    #[test]
    fn clear_on_an_ungrown_arena_keeps_the_first_block() {
        let mut arena = ArcArena::with_block_size(8);
        push_run(&mut arena, 0..3);
        arena.clear();
        assert_eq!(arena.size(), 8);
        assert!(arena.pending().is_empty());
    }

    #[test]
    fn empty_runs_are_representable() {
        let mut arena = ArcArena::<StdArc>::with_block_size(4);
        let run = arena.commit_arcs();
        assert!(run.is_empty());
        assert!(arena.arcs(run).is_empty());
    }

    /// Every pushed element is dropped exactly once, whether it was committed,
    /// dropped mid-run, or still live at teardown. Matters because arcs can carry
    /// owning weights (`StringWeight`, `SparseTupleWeight`, ...).
    #[test]
    fn elements_are_dropped_exactly_once() {
        struct Counted(Rc<Cell<usize>>);
        impl Drop for Counted {
            fn drop(&mut self) {
                self.0.set(self.0.get() + 1);
            }
        }

        let drops = Rc::new(Cell::new(0));
        {
            let mut arena = ArcArena::with_block_size(2);
            // Committed, spanning several blocks.
            for _ in 0..10 {
                arena.push_arc(Counted(Rc::clone(&drops)));
            }
            arena.commit_arcs();
            // Abandoned mid-run.
            for _ in 0..3 {
                arena.push_arc(Counted(Rc::clone(&drops)));
            }
            arena.drop_arcs();
            assert_eq!(drops.get(), 3);

            // Cleared: the committed run goes too.
            arena.clear();
            assert_eq!(drops.get(), 13);

            for _ in 0..5 {
                arena.push_arc(Counted(Rc::clone(&drops)));
            }
            arena.commit_arcs();
        }
        assert_eq!(drops.get(), 18, "arena teardown must drop the live run");
    }
}
