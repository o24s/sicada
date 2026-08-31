//! What both decoders keep: the graph states alive at one frame, and the beam.
//!
//! A frontier maps a graph state to the best cost of reaching it at this frame,
//! plus one `u32` the decoder is free to use: a backpointer in
//! [`viterbi`](crate::viterbi), a lattice state in [`lattice`](crate::lattice).
//! Both decoders prune identically, so the beam lives here rather than in each.

use rustc_hash::FxHashMap;

/// The cost of reaching a graph state at this frame, and one word of whatever
/// the decoder needs alongside it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Token {
    pub cost: f32,
    pub aux: u32,
}

/// Not a valid `aux`, for the decoders that need an absent one.
pub(crate) const NO_AUX: u32 = u32::MAX;

/// How wide to search.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DecodeOptions {
    /// Tokens worse than the frame's best by more than this are dropped.
    ///
    /// In nats, since the scores are. `f32::INFINITY` searches exhaustively,
    /// which the tests compare against.
    pub beam: f32,
    /// A hard cap on the tokens kept per frame, applied after `beam`.
    ///
    /// This is what bounds the work when the beam turns out to be too generous
    /// for a confusable stretch of audio.
    pub max_active: usize,
    /// A floor under the same cap: the beam is never tightened below this many
    /// tokens, so a confident frame does not narrow the search to nothing.
    pub min_active: usize,
}

impl Default for DecodeOptions {
    fn default() -> Self {
        // Kaldi's defaults for a first pass, which are a reasonable starting
        // point for any acoustic model scaled in nats.
        Self {
            beam: 16.0,
            max_active: 7000,
            min_active: 200,
        }
    }
}

impl DecodeOptions {
    /// No beam and no cap: every path is kept.
    ///
    /// The oracle the decoders are tested against searches everything, so this
    /// setting is the one that makes the two comparable.
    pub fn exhaustive() -> Self {
        Self {
            beam: f32::INFINITY,
            max_active: usize::MAX,
            min_active: 0,
        }
    }
}

/// Records `cost` at `state` if it beats what is already there, reporting
/// whether it did.
#[inline]
pub(crate) fn relax_cost<S: std::hash::Hash + Eq>(
    frontier: &mut FxHashMap<S, Token>,
    state: S,
    cost: f32,
    aux: u32,
) -> bool {
    match frontier.get_mut(&state) {
        Some(token) if token.cost <= cost => false,
        Some(token) => {
            *token = Token { cost, aux };
            true
        }
        None => {
            frontier.insert(state, Token { cost, aux });
            true
        }
    }
}

/// Drops what the beam and the cap exclude, and returns the cutoff used.
///
/// `costs` is scratch the caller owns so that pruning allocates nothing per
/// frame.
pub(crate) fn prune<S: std::hash::Hash + Eq>(
    frontier: &mut FxHashMap<S, Token>,
    opts: &DecodeOptions,
    costs: &mut Vec<f32>,
) -> f32 {
    let best = frontier
        .values()
        .map(|token| token.cost)
        .fold(f32::INFINITY, f32::min);
    let mut cutoff = best + opts.beam;

    // The cap is a *tighter* cutoff, found by asking which cost sits at the
    // cap's rank. Selecting is O(n); sorting the frontier would not be.
    let cap = opts.max_active.max(opts.min_active);
    if frontier.len() > cap {
        costs.clear();
        costs.extend(frontier.values().map(|token| token.cost));
        let rank = cap.min(costs.len() - 1);
        let (_, &mut nth, _) = costs.select_nth_unstable_by(rank, |a, b| a.total_cmp(b));
        cutoff = cutoff.min(nth);
    }

    if cutoff.is_finite() {
        frontier.retain(|_, token| token.cost <= cutoff);
    }
    cutoff
}
