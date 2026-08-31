//! Disciplines for choosing which state to visit next.
//!
//! Port of OpenFst's `queue.h`. Search order is the difference between a
//! breadth-first walk, a topological sweep and Dijkstra's algorithm; separating
//! it out lets [`visit`](crate::algorithms::visit) and shortest-distance be
//! written once and then tuned per FST.
//!
//! SICADA-DIVERGE: upstream hangs every discipline off a `QueueBase<S>` virtual
//! base so that `AutoQueue` can hold a `unique_ptr` to whichever it picked. The
//! set of disciplines is closed and known here, so [`AutoQueue`] and
//! [`SccQueue`] dispatch through an enum instead: no allocation per component,
//! and no vtable on the inner loop of shortest-distance.

use std::cell::Cell;
use std::collections::VecDeque;
use std::marker::PhantomData;

use crate::algorithms::cc_visitors::SccVisitor;
use crate::algorithms::dfs_visit::dfs_visit_any;
use crate::algorithms::topsort::TopOrderVisitor;
use crate::arc::{Arc, ArcStateId};
use crate::data_structures::bit_set::GrowableBitSet;
use crate::data_structures::indexed_heap::IndexedHeap;
use crate::fst::Fst;
use crate::properties::{K_ACYCLIC, K_TOP_SORTED, K_UNWEIGHTED};
use crate::weight::{IDEMPOTENT, PATH, Weight};

/// Which discipline a queue follows.
///
/// Reported so that a caller assembling queues out of other queues, as
/// [`AutoQueue`] does, can say what it built.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueueType {
    /// At most one state at a time.
    Trivial,
    /// First in, first out.
    Fifo,
    /// Last in, first out.
    Lifo,
    /// Lightest first.
    ShortestFirst,
    /// In topological order.
    TopOrder,
    /// By state ID.
    StateOrder,
    /// One queue per strongly connected component.
    Scc,
    /// Chosen from the FST's shape.
    Auto,
    /// Something else.
    Other,
}

/// A discipline for choosing which state to visit next.
pub trait Queue<S> {
    /// The state at the head, or `None` when there is none.
    fn head(&self) -> Option<S>;

    /// Adds a state.
    fn enqueue(&mut self, state: S);

    /// Removes the state at the head and returns it.
    ///
    /// SICADA-DIVERGE: upstream splits this into `Head()` and a `void
    /// Dequeue()`, so every caller writes the same two-line dance and an
    /// empty queue is a silent `kNoStateId`. Returning the state makes
    /// `while let Some(s) = queue.dequeue()` the natural loop.
    fn dequeue(&mut self) -> Option<S>;

    /// Tells the queue that a state's weight has changed, so that a discipline
    /// ordering by weight can move it.
    ///
    /// Does nothing for the disciplines whose order does not depend on weight.
    fn update(&mut self, _state: S) {}

    /// Whether the queue holds nothing.
    fn is_empty(&self) -> bool {
        self.head().is_none()
    }

    /// Removes every state.
    fn clear(&mut self);

    /// Which discipline this is.
    fn queue_type(&self) -> QueueType {
        QueueType::Other
    }
}

/// A queue holding at most one state.
///
/// For a strongly connected component of one state with no self-loop, where
/// there is never a choice to make.
#[derive(Debug, Clone, Default)]
pub struct TrivialQueue<S> {
    front: Option<S>,
}

impl<S> TrivialQueue<S> {
    /// Creates an empty queue.
    pub fn new() -> Self {
        Self { front: None }
    }
}

impl<S: Copy> Queue<S> for TrivialQueue<S> {
    fn head(&self) -> Option<S> {
        self.front
    }

    fn enqueue(&mut self, state: S) {
        self.front = Some(state);
    }

    fn dequeue(&mut self) -> Option<S> {
        self.front.take()
    }

    fn clear(&mut self) {
        self.front = None;
    }

    fn queue_type(&self) -> QueueType {
        QueueType::Trivial
    }
}

/// First in, first out: a breadth-first walk.
#[derive(Debug, Clone, Default)]
pub struct FifoQueue<S> {
    states: VecDeque<S>,
}

impl<S> FifoQueue<S> {
    /// Creates an empty queue.
    pub fn new() -> Self {
        Self {
            states: VecDeque::new(),
        }
    }
}

impl<S: Copy> Queue<S> for FifoQueue<S> {
    fn head(&self) -> Option<S> {
        self.states.front().copied()
    }

    fn enqueue(&mut self, state: S) {
        self.states.push_back(state);
    }

    fn dequeue(&mut self) -> Option<S> {
        self.states.pop_front()
    }

    fn is_empty(&self) -> bool {
        self.states.is_empty()
    }

    fn clear(&mut self) {
        self.states.clear();
    }

    fn queue_type(&self) -> QueueType {
        QueueType::Fifo
    }
}

/// Last in, first out: a depth-first walk.
#[derive(Debug, Clone, Default)]
pub struct LifoQueue<S> {
    states: Vec<S>,
}

impl<S> LifoQueue<S> {
    /// Creates an empty queue.
    pub fn new() -> Self {
        Self { states: Vec::new() }
    }
}

impl<S: Copy> Queue<S> for LifoQueue<S> {
    fn head(&self) -> Option<S> {
        self.states.last().copied()
    }

    fn enqueue(&mut self, state: S) {
        self.states.push(state);
    }

    fn dequeue(&mut self) -> Option<S> {
        self.states.pop()
    }

    fn is_empty(&self) -> bool {
        self.states.is_empty()
    }

    fn clear(&mut self) {
        self.states.clear();
    }

    fn queue_type(&self) -> QueueType {
        QueueType::Lifo
    }
}

/// Lightest first, by a caller-supplied ordering.
///
/// Backed by [`IndexedHeap`], whose keys are what let a state already in the
/// queue be moved when its weight improves. That is what Dijkstra's algorithm
/// needs and what a plain binary heap cannot do.
///
/// `UPDATE` says whether to track those keys. With it off, [`Queue::update`]
/// does nothing and enqueuing a state already present inserts it a second time,
/// so the heap holds stale duplicates that are simply popped and ignored. That
/// is what [`SccQueue`] uses: one key vector per component, each sized to the
/// largest state ID that component ever sees, is `nscc` vectors' worth of
/// memory to make an ordering heuristic slightly sharper.
pub struct ShortestFirstQueue<S, C, const UPDATE: bool = true> {
    heap: IndexedHeap<S, C>,
    /// Where each state sits in the heap, or `None` if it is not in it. Empty
    /// when `UPDATE` is false.
    keys: Vec<Option<usize>>,
}

impl<S, C> ShortestFirstQueue<S, C, true>
where
    S: ArcStateId,
    C: Fn(&S, &S) -> bool,
{
    /// Orders states by `comp`, which reports whether its first argument comes
    /// first, and tracks them so that [`Queue::update`] can move one.
    pub fn new(comp: C) -> Self {
        Self {
            heap: IndexedHeap::new(comp),
            keys: Vec::new(),
        }
    }
}

impl<S, C> ShortestFirstQueue<S, C, false>
where
    S: ArcStateId,
    C: Fn(&S, &S) -> bool,
{
    /// As [`new`](ShortestFirstQueue::new), but without the key vector, so
    /// [`Queue::update`] does nothing.
    pub fn without_update(comp: C) -> Self {
        Self {
            heap: IndexedHeap::new(comp),
            keys: Vec::new(),
        }
    }
}

impl<S, C, const UPDATE: bool> ShortestFirstQueue<S, C, UPDATE>
where
    S: ArcStateId,
    C: Fn(&S, &S) -> bool,
{
    /// How many states are queued, counting the stale duplicates that `UPDATE`
    /// off leaves behind.
    pub fn len(&self) -> usize {
        self.heap.len()
    }

    /// Whether nothing is queued.
    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }
}

impl<S, C, const UPDATE: bool> Queue<S> for ShortestFirstQueue<S, C, UPDATE>
where
    S: ArcStateId,
    C: Fn(&S, &S) -> bool,
{
    fn head(&self) -> Option<S> {
        self.heap.top().copied()
    }

    fn enqueue(&mut self, state: S) {
        let key = self.heap.insert(state);
        if UPDATE {
            let index = state.as_usize();
            if self.keys.len() <= index {
                self.keys.resize(index + 1, None);
            }
            self.keys[index] = Some(key);
        }
    }

    fn dequeue(&mut self) -> Option<S> {
        let state = self.heap.pop()?;
        if UPDATE {
            self.keys[state.as_usize()] = None;
        }
        Some(state)
    }

    /// Moves `state` to where its new weight puts it, adding it if it had been
    /// taken out.
    fn update(&mut self, state: S) {
        if !UPDATE {
            return;
        }
        match self.keys.get(state.as_usize()).copied().flatten() {
            Some(key) => self.heap.update(key, state),
            None => self.enqueue(state),
        }
    }

    fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    fn clear(&mut self) {
        self.heap.clear();
        self.keys.clear();
    }

    fn queue_type(&self) -> QueueType {
        QueueType::ShortestFirst
    }
}

/// The strict natural order, without the [`IdempotentWeight`] bound.
///
/// [`crate::weight::natural_less`] demands that bound because the relation is
/// only a partial order on an idempotent semiring. The queues below need the
/// same expression on a weight whose semiring is only known at run time, having
/// already checked [`PATH`]; this is that check spelled out.
///
/// [`IdempotentWeight`]: crate::weight::IdempotentWeight
#[inline]
pub fn natural_less_unchecked<W: Weight>(lhs: &W, rhs: &W) -> bool {
    lhs != rhs && lhs.plus(rhs) == *lhs
}

/// Orders states by a distance vector, under a caller-supplied order on
/// weights.
///
/// The comparison reads the distances as they stand, which is why a state whose
/// distance improves has to be given to [`Queue::update`]: the queue cannot see
/// the change by itself.
///
/// SICADA-DIVERGE: upstream's `StateWeightCompare` holds a bare reference to
/// the distance vector, which the caller is then obliged to keep alive and
/// unmoved for as long as the queue lives, and to mutate through a second handle
/// while the queue reads it. The vector is meant to be shared, so it is shared
/// explicitly.
pub fn state_weight_compare<S, W, L>(
    distance: std::rc::Rc<std::cell::RefCell<Vec<W>>>,
    less: L,
) -> impl Fn(&S, &S) -> bool + Clone
where
    S: ArcStateId,
    W: Weight,
    L: Fn(&W, &W) -> bool + Clone,
{
    move |x: &S, y: &S| {
        let distance = distance.borrow();
        match (distance.get(x.as_usize()), distance.get(y.as_usize())) {
            (Some(wx), Some(wy)) => less(wx, wy),
            // SICADA-DIVERGE: upstream indexes the vector unconditionally, so a
            // state enqueued before its distance was recorded reads past the
            // end. A state with no distance yet sorts last: nothing is known
            // about it, so nothing can be claimed to be worse.
            (Some(_), None) => true,
            _ => false,
        }
    }
}

/// Orders states by a distance vector under the weight's natural order:
/// lightest first.
///
/// Requires an idempotent semiring, without which the natural order is not an
/// order at all.
pub fn natural_state_order<S, W>(
    distance: std::rc::Rc<std::cell::RefCell<Vec<W>>>,
) -> impl Fn(&S, &S) -> bool + Clone
where
    S: ArcStateId,
    W: crate::weight::IdempotentWeight,
{
    state_weight_compare(distance, |a: &W, b: &W| crate::weight::natural_less(a, b))
}

/// An estimate of the remaining distance from a state to a final state, turning
/// Dijkstra's algorithm into A\*.
///
/// An estimate of [`Weight::one`] everywhere is admissible and yields plain
/// Dijkstra, which is [`trivial_estimate`].
pub fn trivial_estimate<S, W: Weight>() -> impl Fn(&S) -> W + Clone {
    |_: &S| W::one()
}

/// An estimate read from a vector of shortest distances to a final state.
///
/// A state past the end of the vector cannot reach a final state, so its
/// estimate is [`Weight::zero`].
pub fn distance_estimate<S, W>(
    beta: std::rc::Rc<std::cell::RefCell<Vec<W>>>,
) -> impl Fn(&S) -> W + Clone
where
    S: ArcStateId,
    W: Weight,
{
    move |s: &S| {
        beta.borrow()
            .get(s.as_usize())
            .cloned()
            .unwrap_or_else(W::zero)
    }
}

/// Orders states by distance-so-far ⊗ estimate-of-what-remains, which makes a
/// [`ShortestFirstQueue`] run A\* rather than Dijkstra.
///
/// The estimate must be admissible, meaning never heavier than the true
/// remaining distance, or the first final state reached need not be the
/// closest.
pub fn a_star_compare<S, W, L, E>(
    distance: std::rc::Rc<std::cell::RefCell<Vec<W>>>,
    less: L,
    estimate: E,
) -> impl Fn(&S, &S) -> bool + Clone
where
    S: ArcStateId,
    W: Weight,
    L: Fn(&W, &W) -> bool + Clone,
    E: Fn(&S) -> W + Clone,
{
    move |x: &S, y: &S| {
        let distance = distance.borrow();
        match (distance.get(x.as_usize()), distance.get(y.as_usize())) {
            (Some(wx), Some(wy)) => less(&wx.times(&estimate(x)), &wy.times(&estimate(y))),
            (Some(_), None) => true,
            _ => false,
        }
    }
}

/// Lightest first, dropping paths that have taken far more arcs than the best
/// path seen.
///
/// In a shortest-path search over a lattice, an old non-viable path can sit in
/// the queue at the same weight as a young promising one. The example upstream
/// gives is a path of 500 arcs costing 10 alongside one of 40 arcs also costing
/// 10, where the short one is very unlikely to lead anywhere good. Counting arcs
/// separates them.
///
/// This relies on the caller exploring shortest-first, dequeuing the head and
/// immediately enqueuing its successors, since that is how a state's arc count
/// is guessed: one more than the head's.
pub struct PruneShortestFirstQueue<S, C, W> {
    inner: ShortestFirstQueue<S, C>,
    /// Arcs on the lightest known path from the start to each state.
    steps: Vec<usize>,
    /// How far behind the longest path a state may be and still be kept.
    /// `None` keeps everything.
    arc_threshold: Option<usize>,
    /// Once more than this many states are queued, the threshold tightens.
    /// `None` never tightens it.
    state_limit: Option<usize>,
    /// Arcs to the current head; read by `head`, hence the cells.
    head_steps: Cell<usize>,
    /// The most arcs any path taken from the head has needed.
    max_head_steps: Cell<usize>,
    _marker: PhantomData<W>,
}

impl<S, C, W> PruneShortestFirstQueue<S, C, W>
where
    S: ArcStateId,
    C: Fn(&S, &S) -> bool,
    W: Weight,
{
    /// Keeps a state only if it is within `arc_threshold` arcs of the longest
    /// path taken so far, tightening that once more than `state_limit` states
    /// are queued.
    pub fn new(comp: C, arc_threshold: Option<usize>, state_limit: Option<usize>) -> Self {
        Self {
            inner: ShortestFirstQueue::new(comp),
            steps: Vec::new(),
            arc_threshold,
            state_limit,
            head_steps: Cell::new(0),
            max_head_steps: Cell::new(0),
            _marker: PhantomData,
        }
    }

    /// How many states are queued.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Whether nothing is queued.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

impl<S, C, W> Queue<S> for PruneShortestFirstQueue<S, C, W>
where
    S: ArcStateId,
    C: Fn(&S, &S) -> bool,
    W: Weight,
{
    fn head(&self) -> Option<S> {
        let head = self.inner.head()?;
        if let Some(&steps) = self.steps.get(head.as_usize()) {
            self.max_head_steps
                .set(self.max_head_steps.get().max(steps));
            self.head_steps.set(steps);
        }
        Some(head)
    }

    fn enqueue(&mut self, state: S) {
        // Assumes an arc from the current head to `state`.
        let state_steps = self.head_steps.get() + 1;
        let index = state.as_usize();
        if index >= self.steps.len() {
            self.steps.resize(index + 1, state_steps);
        }
        self.steps[index] = state_steps;

        let Some(arc_threshold) = self.arc_threshold else {
            self.inner.enqueue(state);
            return;
        };
        // Where counting arcs was not enough to keep the queue small, tighten.
        let adjusted = match self.state_limit {
            Some(limit) if limit > 0 && self.inner.len() > limit => {
                arc_threshold.saturating_sub(self.inner.len() / limit + 1)
            }
            _ => arc_threshold,
        };
        if state_steps > self.max_head_steps.get().saturating_sub(adjusted) {
            if adjusted == 0 && self.state_limit.is_some_and(|limit| limit > 0) {
                // The queue is growing without bound: follow whatever is making
                // progress and drop the rest.
                self.inner.clear();
            }
            self.inner.enqueue(state);
        }
    }

    fn dequeue(&mut self) -> Option<S> {
        // Refresh the step counters, which `head` is responsible for.
        self.head()?;
        self.inner.dequeue()
    }

    fn update(&mut self, state: S) {
        self.inner.update(state);
    }

    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    fn clear(&mut self) {
        self.inner.clear();
        self.steps.clear();
        self.head_steps.set(0);
        self.max_head_steps.set(0);
    }

    fn queue_type(&self) -> QueueType {
        QueueType::ShortestFirst
    }
}

/// Visits states in topological order.
///
/// Only meaningful on an acyclic FST, which is where a single sweep in this
/// order settles every distance.
pub struct TopOrderQueue<S> {
    /// The position each state takes in the order.
    order: Vec<S>,
    /// The state at each position, when it is queued.
    at: Vec<Option<S>>,
    front: usize,
    /// One past the last position holding a state, so an empty queue has
    /// `back == front`.
    back: usize,
}

impl<S: ArcStateId> TopOrderQueue<S> {
    /// Uses a caller-supplied order, where `order[s]` is the position state `s`
    /// takes.
    pub fn with_order(order: Vec<S>) -> Self {
        let len = order.len();
        Self {
            order,
            at: vec![None; len],
            front: 0,
            back: 0,
        }
    }

    /// Works the order out from `fst`, which must be acyclic.
    ///
    /// SICADA-DIVERGE: upstream sets an error flag on a cyclic FST and carries
    /// on with an order vector that was never filled in, so the queue then
    /// indexes past the end of it on the first `Enqueue`. Returning `None`
    /// makes the caller choose another discipline, as [`AutoQueue`] does.
    pub fn new<A, F>(fst: &F) -> Option<Self>
    where
        A: Arc<StateId = S>,
        F: Fst<A>,
    {
        let mut visitor = TopOrderVisitor::<A>::new();
        dfs_visit_any(fst, &mut visitor);
        visitor.order().map(Self::with_order)
    }
}

impl<S: ArcStateId> Queue<S> for TopOrderQueue<S> {
    fn head(&self) -> Option<S> {
        if self.front < self.back {
            self.at[self.front]
        } else {
            None
        }
    }

    fn enqueue(&mut self, state: S) {
        let position = self.order[state.as_usize()].as_usize();
        if self.front >= self.back {
            self.front = position;
            self.back = position + 1;
        } else {
            self.front = self.front.min(position);
            self.back = self.back.max(position + 1);
        }
        self.at[position] = Some(state);
    }

    fn dequeue(&mut self) -> Option<S> {
        let state = self.head()?;
        self.at[self.front] = None;
        while self.front < self.back && self.at[self.front].is_none() {
            self.front += 1;
        }
        Some(state)
    }

    fn is_empty(&self) -> bool {
        self.front >= self.back
    }

    fn clear(&mut self) {
        let back = self.back.min(self.at.len());
        for slot in &mut self.at[self.front.min(back)..back] {
            *slot = None;
        }
        self.front = 0;
        self.back = 0;
    }

    fn queue_type(&self) -> QueueType {
        QueueType::TopOrder
    }
}

/// Visits states in order of their IDs.
///
/// The right discipline for an FST that is already topologically sorted, since
/// then the state order *is* the topological order and no search is needed to
/// find it.
#[derive(Debug, Clone, Default)]
pub struct StateOrderQueue<S> {
    queued: GrowableBitSet,
    front: usize,
    back: usize,
    _marker: PhantomData<S>,
}

impl<S> StateOrderQueue<S> {
    /// Creates an empty queue.
    pub fn new() -> Self {
        Self {
            queued: GrowableBitSet::new(),
            front: 0,
            back: 0,
            _marker: PhantomData,
        }
    }
}

impl<S: ArcStateId> Queue<S> for StateOrderQueue<S> {
    fn head(&self) -> Option<S> {
        (self.front < self.back).then(|| S::from_usize(self.front))
    }

    fn enqueue(&mut self, state: S) {
        let index = state.as_usize();
        if self.front >= self.back {
            self.front = index;
            self.back = index + 1;
        } else {
            self.front = self.front.min(index);
            self.back = self.back.max(index + 1);
        }
        self.queued.insert(index);
    }

    fn dequeue(&mut self) -> Option<S> {
        let state = self.head()?;
        self.queued.remove(self.front);
        while self.front < self.back && !self.queued.contains(self.front) {
            self.front += 1;
        }
        Some(state)
    }

    fn is_empty(&self) -> bool {
        self.front >= self.back
    }

    fn clear(&mut self) {
        for index in self.front..self.back {
            self.queued.remove(index);
        }
        self.front = 0;
        self.back = 0;
    }

    fn queue_type(&self) -> QueueType {
        QueueType::StateOrder
    }
}

/// One of the disciplines [`SccQueue`] runs inside a component.
///
/// SICADA-DIVERGE: upstream stores `nullptr` for a trivial component and keeps
/// the one state in a `trivial_queue_` vector parallel to the queue vector,
/// because a `unique_ptr<QueueBase>` cannot be "a single state". A variant
/// holds it in the component's own slot instead, and the null checks strewn
/// through upstream's `Head`/`Enqueue`/`Dequeue`/`Empty`/`Clear` go away.
pub enum SccInnerQueue<S: ArcStateId, C> {
    /// One state, for a component of one state with no self-loop.
    Trivial(TrivialQueue<S>),
    /// Breadth-first, for a component reached through an arc lighter than
    /// [`Weight::one`], where lightest-first is not valid.
    Fifo(FifoQueue<S>),
    /// Depth-first, for an unweighted component over an idempotent semiring.
    Lifo(LifoQueue<S>),
    /// Lightest first, for a weighted component.
    ShortestFirst(ShortestFirstQueue<S, C, false>),
}

impl<S: ArcStateId, C: Fn(&S, &S) -> bool> Queue<S> for SccInnerQueue<S, C> {
    fn head(&self) -> Option<S> {
        match self {
            Self::Trivial(q) => q.head(),
            Self::Fifo(q) => q.head(),
            Self::Lifo(q) => q.head(),
            Self::ShortestFirst(q) => q.head(),
        }
    }

    fn enqueue(&mut self, state: S) {
        match self {
            Self::Trivial(q) => q.enqueue(state),
            Self::Fifo(q) => q.enqueue(state),
            Self::Lifo(q) => q.enqueue(state),
            Self::ShortestFirst(q) => q.enqueue(state),
        }
    }

    fn dequeue(&mut self) -> Option<S> {
        match self {
            Self::Trivial(q) => q.dequeue(),
            Self::Fifo(q) => q.dequeue(),
            Self::Lifo(q) => q.dequeue(),
            Self::ShortestFirst(q) => q.dequeue(),
        }
    }

    fn update(&mut self, state: S) {
        match self {
            Self::Trivial(q) => q.update(state),
            Self::Fifo(q) => q.update(state),
            Self::Lifo(q) => q.update(state),
            Self::ShortestFirst(q) => q.update(state),
        }
    }

    fn is_empty(&self) -> bool {
        match self {
            Self::Trivial(q) => Queue::is_empty(q),
            Self::Fifo(q) => Queue::is_empty(q),
            Self::Lifo(q) => Queue::is_empty(q),
            Self::ShortestFirst(q) => Queue::is_empty(q),
        }
    }

    fn clear(&mut self) {
        match self {
            Self::Trivial(q) => q.clear(),
            Self::Fifo(q) => q.clear(),
            Self::Lifo(q) => q.clear(),
            Self::ShortestFirst(q) => q.clear(),
        }
    }

    fn queue_type(&self) -> QueueType {
        match self {
            Self::Trivial(_) => QueueType::Trivial,
            Self::Fifo(_) => QueueType::Fifo,
            Self::Lifo(_) => QueueType::Lifo,
            Self::ShortestFirst(_) => QueueType::ShortestFirst,
        }
    }
}

/// Visits strongly connected components in topological order, running its own
/// discipline inside each.
///
/// Distances within a component have to reach a fixed point before any component
/// downstream of it can be settled, so there is nothing to gain from
/// interleaving them and a great deal to lose: a global lightest-first queue
/// pays `log n` on every operation over the whole FST rather than over one
/// component.
pub struct SccQueue<S: ArcStateId, C> {
    queues: Vec<SccInnerQueue<S, C>>,
    /// The component each state belongs to, numbered in topological order.
    scc: Vec<S>,
    /// The component being drained; advanced past empty ones by `head`, hence
    /// the cell.
    front: Cell<usize>,
    /// One past the last component holding a state.
    back: usize,
}

impl<S: ArcStateId, C: Fn(&S, &S) -> bool> SccQueue<S, C> {
    /// Takes the component of each state and the discipline to run in each
    /// component.
    pub fn new(scc: Vec<S>, queues: Vec<SccInnerQueue<S, C>>) -> Self {
        Self {
            queues,
            scc,
            front: Cell::new(0),
            back: 0,
        }
    }

    /// Advances past the components that have been drained and reports the
    /// first that still holds a state.
    fn skip_empty(&self) -> Option<usize> {
        let mut front = self.front.get();
        while front < self.back && self.queues[front].is_empty() {
            front += 1;
        }
        self.front.set(front);
        (front < self.back).then_some(front)
    }

    /// The discipline chosen for each component.
    pub fn inner_types(&self) -> Vec<QueueType> {
        self.queues.iter().map(|q| q.queue_type()).collect()
    }
}

impl<S: ArcStateId, C: Fn(&S, &S) -> bool> Queue<S> for SccQueue<S, C> {
    fn head(&self) -> Option<S> {
        self.queues[self.skip_empty()?].head()
    }

    fn enqueue(&mut self, state: S) {
        let scc = self.scc[state.as_usize()].as_usize();
        let front = self.front.get();
        if front >= self.back {
            self.front.set(scc);
            self.back = scc + 1;
        } else {
            self.front.set(front.min(scc));
            self.back = self.back.max(scc + 1);
        }
        self.queues[scc].enqueue(state);
    }

    fn dequeue(&mut self) -> Option<S> {
        let front = self.skip_empty()?;
        self.queues[front].dequeue()
    }

    fn update(&mut self, state: S) {
        let scc = self.scc[state.as_usize()].as_usize();
        self.queues[scc].update(state);
    }

    fn is_empty(&self) -> bool {
        self.skip_empty().is_none()
    }

    fn clear(&mut self) {
        for queue in &mut self.queues[self.front.get().min(self.back)..self.back] {
            queue.clear();
        }
        self.front.set(0);
        self.back = 0;
    }

    fn queue_type(&self) -> QueueType {
        QueueType::Scc
    }
}

/// What [`scc_queue_types`] found out about an FST's components.
#[derive(Debug, Clone)]
pub struct SccAnalysis {
    /// The discipline to run inside each component.
    pub queue_types: Vec<QueueType>,
    /// Whether every component is trivial, which means the FST is acyclic and
    /// the component numbers already give a topological order.
    pub all_trivial: bool,
    /// Whether the semiring is idempotent and every arc weight is
    /// [`Weight::zero`] or [`Weight::one`], which means any order will settle
    /// the distances.
    pub unweighted: bool,
}

/// Decides which discipline to run inside each strongly connected component.
///
/// A component is trivial until an arc is found that stays inside it. Such an
/// arc makes the component cyclic, and then:
///
/// - without a weight order to go on, or with an arc lighter than
///   [`Weight::one`], where lightest-first is not valid because extending a path
///   could improve it, the component gets a FIFO queue;
/// - with every arc weight [`Weight::zero`] or [`Weight::one`] over an
///   idempotent semiring, LIFO, which is cheaper and just as good;
/// - otherwise lightest-first.
pub fn scc_queue_types<A, F, L>(
    fst: &F,
    scc: &[A::StateId],
    nscc: usize,
    less: Option<&L>,
) -> SccAnalysis
where
    A: Arc,
    F: Fst<A>,
    L: Fn(&A::Weight, &A::Weight) -> bool,
{
    let mut queue_types = vec![QueueType::Trivial; nscc];
    let mut all_trivial = true;
    let idempotent = A::Weight::properties() & IDEMPOTENT != 0;
    let mut unweighted = idempotent;
    let (zero, one) = (A::Weight::zero(), A::Weight::one());

    for state in fst.states() {
        let Some(&from) = scc.get(state.as_usize()) else {
            continue;
        };
        for arc in fst.arcs(state) {
            let plain = idempotent && (*arc.weight() == zero || *arc.weight() == one);
            if scc.get(arc.nextstate().as_usize()) == Some(&from) {
                let ty = &mut queue_types[from.as_usize()];
                match less {
                    // No weight order to go on: nothing better than FIFO.
                    None => *ty = QueueType::Fifo,
                    // An arc lighter than One means a longer path can be
                    // lighter, so lightest-first would settle a state too soon.
                    Some(less) if less(arc.weight(), &one) => *ty = QueueType::Fifo,
                    Some(_) if matches!(*ty, QueueType::Trivial | QueueType::Lifo) => {
                        *ty = if plain {
                            QueueType::Lifo
                        } else {
                            QueueType::ShortestFirst
                        };
                    }
                    Some(_) => {}
                }
                if *ty != QueueType::Trivial {
                    all_trivial = false;
                }
            }
            if !plain {
                unweighted = false;
            }
        }
    }

    SccAnalysis {
        queue_types,
        all_trivial,
        unweighted,
    }
}

/// The strongly connected component each state belongs to, numbered in
/// topological order.
pub fn components<A: Arc, F: Fst<A>>(fst: &F) -> Vec<A::StateId> {
    let mut scc = Vec::new();
    let mut props = 0;
    let mut visitor =
        SccVisitor::new(fst, Some(&mut scc), None, None, &mut props).without_coaccess();
    dfs_visit_any(fst, &mut visitor);
    drop(visitor);
    scc
}

/// Picks a discipline from the shape of the FST.
///
/// The order that settles distances soonest depends on the FST: a topologically
/// sorted one needs no search at all, an acyclic one needs one topological
/// sweep, an unweighted one over an idempotent semiring can be walked in any
/// order, and anything else is decomposed into strongly connected components
/// and given a discipline per component.
pub enum AutoQueue<S: ArcStateId, C> {
    /// The FST is already sorted, so state order will do.
    StateOrder(StateOrderQueue<S>),
    /// The FST is acyclic, so one topological sweep will do.
    TopOrder(TopOrderQueue<S>),
    /// Unweighted over an idempotent semiring: any order settles it.
    Lifo(LifoQueue<S>),
    /// Cyclic and weighted: a discipline per component.
    Scc(SccQueue<S, C>),
}

impl<S: ArcStateId, C: Fn(&S, &S) -> bool + Clone> AutoQueue<S, C> {
    /// Chooses a discipline for `fst`.
    ///
    /// `comp` orders states by weight, for the components that need it; pass
    /// `None` when there is no distance vector to order by, and those
    /// components fall back to FIFO.
    pub fn new<A, F>(fst: &F, comp: Option<C>) -> Self
    where
        A: Arc<StateId = S>,
        F: Fst<A>,
    {
        let props = fst.properties(K_ACYCLIC | K_TOP_SORTED | K_UNWEIGHTED, false);
        if props & K_TOP_SORTED != 0 || fst.start().is_none() {
            return Self::StateOrder(StateOrderQueue::new());
        }
        if props & K_ACYCLIC != 0
            && let Some(queue) = TopOrderQueue::new(fst)
        {
            return Self::TopOrder(queue);
        }
        let idempotent = A::Weight::properties() & IDEMPOTENT != 0;
        if props & K_UNWEIGHTED != 0 && idempotent {
            return Self::Lifo(LifoQueue::new());
        }

        let scc = components(fst);
        let Some(nscc) = scc.iter().map(|s| s.as_usize() + 1).max() else {
            return Self::StateOrder(StateOrderQueue::new());
        };

        // The natural order is only an order on a semiring with the path
        // property; without it there is nothing to compare arc weights with,
        // and every cyclic component gets FIFO.
        let less = (A::Weight::properties() & PATH != 0 && comp.is_some())
            .then_some(natural_less_unchecked::<A::Weight>);
        let analysis = scc_queue_types(fst, &scc, nscc, less.as_ref());

        if analysis.unweighted {
            return Self::Lifo(LifoQueue::new());
        }
        if analysis.all_trivial {
            // Every component is a single state with no self-loop, so the FST
            // is acyclic after all and the component numbers are already a
            // topological order.
            return Self::TopOrder(TopOrderQueue::with_order(scc));
        }

        let queues = analysis
            .queue_types
            .iter()
            .map(|ty| match ty {
                QueueType::Trivial => SccInnerQueue::Trivial(TrivialQueue::new()),
                QueueType::Lifo => SccInnerQueue::Lifo(LifoQueue::new()),
                QueueType::ShortestFirst => match comp.clone() {
                    Some(comp) => {
                        SccInnerQueue::ShortestFirst(ShortestFirstQueue::without_update(comp))
                    }
                    // scc_queue_types only says ShortestFirst when it was given
                    // an order, which it only is when `comp` is present.
                    None => SccInnerQueue::Fifo(FifoQueue::new()),
                },
                _ => SccInnerQueue::Fifo(FifoQueue::new()),
            })
            .collect();
        Self::Scc(SccQueue::new(scc, queues))
    }

    /// Which discipline was chosen.
    pub fn chosen(&self) -> QueueType {
        match self {
            Self::StateOrder(_) => QueueType::StateOrder,
            Self::TopOrder(_) => QueueType::TopOrder,
            Self::Lifo(_) => QueueType::Lifo,
            Self::Scc(_) => QueueType::Scc,
        }
    }

    /// The per-component disciplines, when one was chosen per component.
    pub fn inner_types(&self) -> Option<Vec<QueueType>> {
        match self {
            Self::Scc(queue) => Some(queue.inner_types()),
            _ => None,
        }
    }
}

impl<S: ArcStateId, C: Fn(&S, &S) -> bool> Queue<S> for AutoQueue<S, C> {
    fn head(&self) -> Option<S> {
        match self {
            Self::StateOrder(q) => q.head(),
            Self::TopOrder(q) => q.head(),
            Self::Lifo(q) => q.head(),
            Self::Scc(q) => q.head(),
        }
    }

    fn enqueue(&mut self, state: S) {
        match self {
            Self::StateOrder(q) => q.enqueue(state),
            Self::TopOrder(q) => q.enqueue(state),
            Self::Lifo(q) => q.enqueue(state),
            Self::Scc(q) => q.enqueue(state),
        }
    }

    fn dequeue(&mut self) -> Option<S> {
        match self {
            Self::StateOrder(q) => q.dequeue(),
            Self::TopOrder(q) => q.dequeue(),
            Self::Lifo(q) => q.dequeue(),
            Self::Scc(q) => q.dequeue(),
        }
    }

    fn update(&mut self, state: S) {
        match self {
            Self::StateOrder(q) => q.update(state),
            Self::TopOrder(q) => q.update(state),
            Self::Lifo(q) => q.update(state),
            Self::Scc(q) => q.update(state),
        }
    }

    fn is_empty(&self) -> bool {
        match self {
            Self::StateOrder(q) => q.is_empty(),
            Self::TopOrder(q) => q.is_empty(),
            Self::Lifo(q) => Queue::is_empty(q),
            Self::Scc(q) => q.is_empty(),
        }
    }

    fn clear(&mut self) {
        match self {
            Self::StateOrder(q) => q.clear(),
            Self::TopOrder(q) => q.clear(),
            Self::Lifo(q) => q.clear(),
            Self::Scc(q) => q.clear(),
        }
    }

    fn queue_type(&self) -> QueueType {
        QueueType::Auto
    }
}

/// Wraps a queue, refusing states whose distance is worse than a threshold
/// beyond the best in their equivalence class.
///
/// The class function is what gives this its use: with every state in its own
/// class it prunes against that state's own best distance, and with states
/// grouped, by the lattice position they represent for instance, it prunes a
/// state against its peers.
pub struct PruneQueue<S, Q, W, L, F> {
    distance: std::rc::Rc<std::cell::RefCell<Vec<W>>>,
    queue: Q,
    less: L,
    class_fnc: F,
    threshold: W,
    /// The best distance seen in each class.
    class_distance: Vec<W>,
    _marker: PhantomData<S>,
}

impl<S, Q, W, L, F> PruneQueue<S, Q, W, L, F>
where
    S: ArcStateId,
    Q: Queue<S>,
    W: Weight,
    L: Fn(&W, &W) -> bool,
    F: Fn(S) -> usize,
{
    /// Prunes anything not within `threshold` of the best distance in its
    /// class, under the order `less`.
    pub fn new(
        distance: std::rc::Rc<std::cell::RefCell<Vec<W>>>,
        queue: Q,
        less: L,
        class_fnc: F,
        threshold: W,
    ) -> Self {
        Self {
            distance,
            queue,
            less,
            class_fnc,
            threshold,
            class_distance: Vec::new(),
            _marker: PhantomData,
        }
    }

    /// The queue underneath.
    pub fn inner(&self) -> &Q {
        &self.queue
    }

    /// Records `state`'s distance against its class and reports that distance.
    fn note(&mut self, state: S) -> Option<W> {
        let class = (self.class_fnc)(state);
        if class >= self.class_distance.len() {
            self.class_distance.resize(class + 1, W::zero());
        }
        let distance = self.distance.borrow().get(state.as_usize()).cloned()?;
        if (self.less)(&distance, &self.class_distance[class]) {
            self.class_distance[class] = distance.clone();
        }
        Some(distance)
    }
}

impl<S, Q, W, L, F> Queue<S> for PruneQueue<S, Q, W, L, F>
where
    S: ArcStateId,
    Q: Queue<S>,
    W: Weight,
    L: Fn(&W, &W) -> bool,
    F: Fn(S) -> usize,
{
    fn head(&self) -> Option<S> {
        self.queue.head()
    }

    fn enqueue(&mut self, state: S) {
        let Some(distance) = self.note(state) else {
            return;
        };
        let class = (self.class_fnc)(state);
        let limit = self.class_distance[class].times(&self.threshold);
        if (self.less)(&distance, &limit) {
            self.queue.enqueue(state);
        }
    }

    fn dequeue(&mut self) -> Option<S> {
        self.queue.dequeue()
    }

    fn update(&mut self, state: S) {
        self.note(state);
        self.queue.update(state);
    }

    fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    fn clear(&mut self) {
        self.queue.clear();
    }
}

impl<S, Q, W, F> PruneQueue<S, Q, W, fn(&W, &W) -> bool, F>
where
    S: ArcStateId,
    Q: Queue<S>,
    W: crate::weight::IdempotentWeight,
    F: Fn(S) -> usize,
{
    /// As [`new`](PruneQueue::new), under the weight's natural order.
    pub fn natural(
        distance: std::rc::Rc<std::cell::RefCell<Vec<W>>>,
        queue: Q,
        class_fnc: F,
        threshold: W,
    ) -> Self {
        Self::new(
            distance,
            queue,
            crate::weight::natural_less::<W> as fn(&W, &W) -> bool,
            class_fnc,
            threshold,
        )
    }
}

/// Wraps a queue, refusing the states a filter rejects.
pub struct FilterQueue<S, Q, F> {
    queue: Q,
    filter: F,
    _marker: PhantomData<S>,
}

impl<S, Q, F> FilterQueue<S, Q, F>
where
    S: ArcStateId,
    Q: Queue<S>,
    F: Fn(S) -> bool,
{
    /// Enqueues a state only when `filter` accepts it.
    pub fn new(queue: Q, filter: F) -> Self {
        Self {
            queue,
            filter,
            _marker: PhantomData,
        }
    }

    /// The queue underneath.
    pub fn inner(&self) -> &Q {
        &self.queue
    }
}

impl<S, Q, F> Queue<S> for FilterQueue<S, Q, F>
where
    S: ArcStateId,
    Q: Queue<S>,
    F: Fn(S) -> bool,
{
    fn head(&self) -> Option<S> {
        self.queue.head()
    }

    fn enqueue(&mut self, state: S) {
        if (self.filter)(state) {
            self.queue.enqueue(state);
        }
    }

    fn dequeue(&mut self) -> Option<S> {
        self.queue.dequeue()
    }

    fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    fn clear(&mut self) {
        self.queue.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arc::StdArc;
    use crate::fst::{ExpandedFst as _, MutableFst};
    use crate::fsts::vector_fst::StdVectorFst;
    use crate::properties::K_FST_PROPERTIES;
    use crate::weight::natural_less;
    use crate::weights::float_weight::TropicalWeight;
    use std::cell::RefCell;
    use std::rc::Rc;

    type Distance = Rc<RefCell<Vec<TropicalWeight>>>;

    /// Drains a queue into the order it hands states back.
    fn drain<Q: Queue<i32>>(queue: &mut Q) -> Vec<i32> {
        let mut out = Vec::new();
        while let Some(state) = queue.dequeue() {
            out.push(state);
        }
        out
    }

    fn distance_of(weights: &[f32]) -> Distance {
        Rc::new(RefCell::new(
            weights.iter().copied().map(TropicalWeight).collect(),
        ))
    }

    fn natural(distance: &Distance) -> impl Fn(&i32, &i32) -> bool + Clone + use<> {
        natural_state_order::<i32, TropicalWeight>(Rc::clone(distance))
    }

    #[test]
    fn fifo_and_lifo_are_the_two_obvious_orders() {
        let mut fifo = FifoQueue::new();
        let mut lifo = LifoQueue::new();
        for s in [3, 1, 2] {
            fifo.enqueue(s);
            lifo.enqueue(s);
        }
        assert_eq!(drain(&mut fifo), vec![3, 1, 2]);
        assert_eq!(drain(&mut lifo), vec![2, 1, 3]);
    }

    #[test]
    fn a_trivial_queue_holds_one_state() {
        let mut queue = TrivialQueue::new();
        assert!(Queue::is_empty(&queue));
        queue.enqueue(5);
        assert_eq!(queue.head(), Some(5));
        queue.enqueue(7);
        assert_eq!(queue.head(), Some(7), "the second replaces the first");
        assert_eq!(drain(&mut queue), vec![7]);
    }

    #[test]
    fn state_order_hands_states_back_by_id() {
        let mut queue = StateOrderQueue::new();
        for s in [4, 1, 7, 2] {
            queue.enqueue(s);
        }
        assert_eq!(drain(&mut queue), vec![1, 2, 4, 7]);
    }

    /// Enqueuing a state already queued does not queue it twice.
    #[test]
    fn state_order_holds_each_state_once() {
        let mut queue = StateOrderQueue::new();
        for s in [3, 1, 3, 1, 3] {
            queue.enqueue(s);
        }
        assert_eq!(drain(&mut queue), vec![1, 3]);
    }

    #[test]
    fn shortest_first_hands_back_the_lightest() {
        let distance = distance_of(&[5.0, 2.0, 9.0, 1.0]);
        let mut queue = ShortestFirstQueue::new(natural(&distance));
        for s in 0..4 {
            queue.enqueue(s);
        }
        assert_eq!(drain(&mut queue), vec![3, 1, 0, 2]);
    }

    /// The reason the queue is indexed: a state's weight can improve while it
    /// is waiting, and it has to move.
    #[test]
    fn shortest_first_moves_a_state_whose_weight_improves() {
        let distance = distance_of(&[5.0, 2.0, 9.0]);
        let mut queue = ShortestFirstQueue::new(natural(&distance));
        for s in 0..3 {
            queue.enqueue(s);
        }

        // State 2 was the heaviest; now it is the lightest.
        distance.borrow_mut()[2] = TropicalWeight(0.5);
        queue.update(2);
        assert_eq!(drain(&mut queue), vec![2, 1, 0]);
    }

    /// A state taken out and then improved is put back.
    #[test]
    fn updating_a_state_that_is_not_queued_adds_it() {
        let distance = distance_of(&[1.0, 2.0]);
        let mut queue = ShortestFirstQueue::new(natural(&distance));
        queue.enqueue(0);
        assert_eq!(queue.dequeue(), Some(0));
        assert!(Queue::is_empty(&queue));

        queue.update(1);
        assert_eq!(drain(&mut queue), vec![1]);
    }

    /// Without key tracking, `update` cannot move anything and enqueuing twice
    /// leaves a stale duplicate, which is the trade [`SccQueue`] takes.
    #[test]
    fn without_key_tracking_a_state_cannot_be_moved() {
        let distance = distance_of(&[5.0, 2.0, 9.0]);
        let mut tracking = ShortestFirstQueue::new(natural(&distance));
        let mut untracking = ShortestFirstQueue::without_update(natural(&distance));
        for s in 0..3 {
            tracking.enqueue(s);
            untracking.enqueue(s);
        }
        assert_eq!(tracking.head(), Some(1));
        assert_eq!(untracking.head(), Some(1));

        // State 2 was the heaviest; now it is the lightest.
        distance.borrow_mut()[2] = TropicalWeight(0.5);
        tracking.update(2);
        untracking.update(2);
        assert_eq!(tracking.head(), Some(2), "the improvement moved it");
        assert_eq!(
            untracking.head(),
            Some(1),
            "nothing knows where state 2 sits, so it stays where its old weight put it"
        );
    }

    /// 0 → 2 → 1, so topological order is 0, 2, 1 while state order is 0, 1, 2.
    fn zigzag() -> StdVectorFst {
        let mut fst = StdVectorFst::new();
        for _ in 0..3 {
            fst.add_state();
        }
        fst.set_start(0);
        fst.add_arc(0, StdArc::new(1, 1, TropicalWeight::one(), 2));
        fst.add_arc(2, StdArc::new(1, 1, TropicalWeight::one(), 1));
        fst.set_final(1, TropicalWeight::one());
        fst.properties(K_FST_PROPERTIES, true);
        fst
    }

    #[test]
    fn top_order_follows_the_arcs_rather_than_the_numbering() {
        let fst = zigzag();
        let mut queue = TopOrderQueue::new(&fst).expect("acyclic");
        for s in 0..3 {
            queue.enqueue(s);
        }
        assert_eq!(drain(&mut queue), vec![0, 2, 1]);
    }

    #[test]
    fn there_is_no_topological_order_for_a_cyclic_fst() {
        let mut fst = zigzag();
        fst.add_arc(1, StdArc::new(1, 1, TropicalWeight::one(), 0));
        assert!(TopOrderQueue::<i32>::new(&fst).is_none());
    }

    /// Two two-state cycles in a row: 0↔1 → 2↔3.
    fn two_components() -> StdVectorFst {
        let mut fst = StdVectorFst::new();
        for _ in 0..4 {
            fst.add_state();
        }
        fst.set_start(0);
        fst.add_arc(0, StdArc::new(1, 1, TropicalWeight(1.0), 1));
        fst.add_arc(1, StdArc::new(1, 1, TropicalWeight(1.0), 0));
        fst.add_arc(1, StdArc::new(2, 2, TropicalWeight(1.0), 2));
        fst.add_arc(2, StdArc::new(1, 1, TropicalWeight(1.0), 3));
        fst.add_arc(3, StdArc::new(1, 1, TropicalWeight(1.0), 2));
        fst.set_final(3, TropicalWeight::one());
        fst.properties(K_FST_PROPERTIES, true);
        fst
    }

    /// The upstream numbering has to come out topological, or the meta-queue
    /// would drain components in the wrong order.
    #[test]
    fn components_are_numbered_in_topological_order() {
        let scc = components(&two_components());
        assert_eq!(scc[0], scc[1]);
        assert_eq!(scc[2], scc[3]);
        assert!(scc[0] < scc[2], "the upstream component comes first");
    }

    /// A cyclic component with real weights wants lightest-first; one with no
    /// internal arc at all stays trivial.
    #[test]
    fn each_component_gets_the_discipline_its_arcs_call_for() {
        let fst = two_components();
        let scc = components(&fst);
        let nscc = scc.iter().map(|s| *s as usize + 1).max().unwrap();
        let less = |a: &TropicalWeight, b: &TropicalWeight| natural_less(a, b);

        let with = scc_queue_types(&fst, &scc, nscc, Some(&less));
        assert_eq!(
            with.queue_types,
            vec![QueueType::ShortestFirst, QueueType::ShortestFirst]
        );
        assert!(!with.all_trivial);
        assert!(!with.unweighted);

        // With nothing to order weights by, there is nothing better than FIFO.
        type Less = fn(&TropicalWeight, &TropicalWeight) -> bool;
        let without = scc_queue_types::<StdArc, _, Less>(&fst, &scc, nscc, None);
        assert_eq!(without.queue_types, vec![QueueType::Fifo, QueueType::Fifo]);

        // An acyclic FST has no arc staying inside a component at all.
        let acyclic = zigzag();
        let scc = components(&acyclic);
        let nscc = scc.iter().map(|s| *s as usize + 1).max().unwrap();
        let trivial = scc_queue_types(&acyclic, &scc, nscc, Some(&less));
        assert!(trivial.all_trivial);
        assert!(
            trivial.queue_types.iter().all(|t| *t == QueueType::Trivial),
            "{:?}",
            trivial.queue_types
        );
    }

    /// The meta-queue drains one component before starting the next, whatever
    /// order states are handed to it in.
    #[test]
    fn the_meta_queue_finishes_a_component_before_moving_on() {
        let fst = two_components();
        let scc = components(&fst);
        let distance = distance_of(&[0.0, 1.0, 2.0, 3.0]);
        let queues = vec![
            SccInnerQueue::ShortestFirst(ShortestFirstQueue::without_update(natural(&distance))),
            SccInnerQueue::ShortestFirst(ShortestFirstQueue::without_update(natural(&distance))),
        ];
        let mut queue = SccQueue::new(scc.clone(), queues);

        // Enqueued downstream-first, to show the order does not come from that.
        for s in [3, 2, 1, 0] {
            queue.enqueue(s);
        }
        let order = drain(&mut queue);
        assert_eq!(order, vec![0, 1, 2, 3]);
        let positions: Vec<usize> = order.iter().map(|s| scc[*s as usize] as usize).collect();
        assert!(
            positions.windows(2).all(|w| w[0] <= w[1]),
            "components have to come out in topological order: {positions:?}"
        );
    }

    /// The choice AutoQueue makes, for each shape it recognizes.
    #[test]
    fn the_automatic_choice_follows_the_shape_of_the_fst() {
        use crate::algorithms::topsort::top_sort;

        let distance = distance_of(&[0.0; 8]);
        let comp = || Some(natural(&distance));

        // Already sorted: state order needs no search at all.
        let mut sorted = zigzag();
        top_sort(&mut sorted).unwrap();
        assert_eq!(
            AutoQueue::new(&sorted, comp()).chosen(),
            QueueType::StateOrder
        );

        // Acyclic but not sorted: one topological sweep.
        assert_eq!(
            AutoQueue::new(&zigzag(), comp()).chosen(),
            QueueType::TopOrder
        );

        // Cyclic and unweighted over an idempotent semiring: any order does.
        let mut cyclic = zigzag();
        cyclic.add_arc(1, StdArc::new(1, 1, TropicalWeight::one(), 0));
        cyclic.properties(K_FST_PROPERTIES, true);
        assert_eq!(AutoQueue::new(&cyclic, comp()).chosen(), QueueType::Lifo);

        // Cyclic and weighted: a discipline per component.
        let weighted = two_components();
        let auto = AutoQueue::new(&weighted, comp());
        assert_eq!(auto.chosen(), QueueType::Scc);
        assert_eq!(
            auto.inner_types(),
            Some(vec![QueueType::ShortestFirst, QueueType::ShortestFirst])
        );
    }

    /// An FST whose properties do not admit acyclicity, but whose components all
    /// turn out to be trivial, is acyclic after all, and the component numbers
    /// are already the topological order, so no second search is run.
    #[test]
    fn components_that_are_all_trivial_give_the_topological_order() {
        let mut fst = zigzag();
        // Hide the acyclic bit so AutoQueue has to work it out from the SCCs.
        fst.set_properties(0, K_ACYCLIC | K_TOP_SORTED | K_UNWEIGHTED);
        fst.add_arc(0, StdArc::new(3, 3, TropicalWeight(2.0), 1));
        fst.set_properties(0, K_ACYCLIC | K_TOP_SORTED | K_UNWEIGHTED);

        let distance = distance_of(&[0.0; 3]);
        let auto = AutoQueue::new(&fst, Some(natural(&distance)));
        assert_eq!(auto.chosen(), QueueType::TopOrder);
    }

    #[test]
    fn clearing_empties_every_discipline() {
        let mut fifo = FifoQueue::new();
        let mut lifo = LifoQueue::new();
        let mut order = StateOrderQueue::new();
        let mut top = TopOrderQueue::new(&zigzag()).unwrap();
        for s in [1, 2, 0] {
            fifo.enqueue(s);
            lifo.enqueue(s);
            order.enqueue(s);
            top.enqueue(s);
        }
        fifo.clear();
        lifo.clear();
        order.clear();
        top.clear();
        assert!(Queue::is_empty(&fifo));
        assert!(Queue::is_empty(&lifo));
        assert!(Queue::is_empty(&order));
        assert!(Queue::is_empty(&top));
        // And they still work afterwards.
        order.enqueue(5);
        assert_eq!(drain(&mut order), vec![5]);
        top.enqueue(1);
        assert_eq!(drain(&mut top), vec![1]);
    }

    #[test]
    fn a_filter_queue_refuses_what_the_filter_rejects() {
        let mut queue = FilterQueue::new(FifoQueue::new(), |s: i32| s % 2 == 0);
        for s in 0..6 {
            queue.enqueue(s);
        }
        assert_eq!(drain(&mut queue), vec![0, 2, 4]);
    }

    /// With every state in its own class, a state is kept only if it is within
    /// the threshold of its own best distance, which it always is, so the
    /// interesting case is states sharing a class.
    #[test]
    fn a_prune_queue_refuses_what_is_far_behind_its_class() {
        let distance = distance_of(&[0.0, 1.0, 5.0, 2.0]);
        // Everything in one class, threshold 2: keep anything under best + 2.
        let mut queue = PruneQueue::natural(
            Rc::clone(&distance),
            FifoQueue::new(),
            |_: i32| 0,
            TropicalWeight(2.0),
        );
        for s in 0..4 {
            queue.enqueue(s);
        }
        // Best is 0 (state 0), so the limit is 2: state 1 is in, state 3 is at
        // the limit and so is not strictly under it, state 2 is far out. State
        // 0 sets the class distance to 0 and is then not under 0 + 2... it is,
        // so it stays.
        assert_eq!(drain(&mut queue), vec![0, 1]);
    }

    /// Counting arcs is what separates an old non-viable path from a young
    /// promising one at the same weight.
    #[test]
    fn the_pruning_queue_drops_a_path_that_has_taken_too_many_arcs() {
        // States 0..=3 are a light chain; state 4 sits one arc from the start
        // on a path so heavy that it is not looked at until the chain is done.
        let distance = distance_of(&[0.0, 1.0, 2.0, 3.0, 100.0, 101.0]);
        let mut queue: PruneShortestFirstQueue<i32, _, TropicalWeight> =
            PruneShortestFirstQueue::new(natural(&distance), Some(1), None);

        queue.enqueue(0);
        queue.enqueue(4);
        for s in 1..4 {
            assert_eq!(queue.dequeue(), Some(s - 1));
            queue.enqueue(s);
        }
        assert_eq!(queue.dequeue(), Some(3), "the chain is four arcs long");

        // State 4 is one arc from the start, so anything reached from it is
        // two arcs in against a four-arc best path.
        assert_eq!(queue.dequeue(), Some(4));
        queue.enqueue(5);
        assert!(
            Queue::is_empty(&queue),
            "two arcs in is three behind the four-arc path, and the threshold is one"
        );
    }

    /// With no threshold nothing is dropped, whatever the arc counts.
    #[test]
    fn the_pruning_queue_keeps_everything_without_a_threshold() {
        let distance = distance_of(&[0.0, 1.0, 2.0, 3.0]);
        let mut queue: PruneShortestFirstQueue<i32, _, TropicalWeight> =
            PruneShortestFirstQueue::new(natural(&distance), None, None);
        for s in 0..4 {
            queue.enqueue(s);
        }
        assert_eq!(drain(&mut queue), vec![0, 1, 2, 3]);
    }

    /// A\* orders by distance-so-far ⊗ estimate-of-what-remains, so a state
    /// that is close now but far from the end sorts behind one that is not.
    #[test]
    fn an_a_star_estimate_reorders_the_queue() {
        let distance = distance_of(&[1.0, 3.0]);
        let plain = ShortestFirstQueue::new(state_weight_compare(
            Rc::clone(&distance),
            |a: &TropicalWeight, b: &TropicalWeight| natural_less(a, b),
        ));
        let mut plain = plain;
        plain.enqueue(0);
        plain.enqueue(1);
        assert_eq!(drain(&mut plain), vec![0, 1]);

        // State 0 needs 10 more, state 1 only 1: total 11 against 4.
        let beta = distance_of(&[10.0, 1.0]);
        let mut astar = ShortestFirstQueue::new(a_star_compare(
            Rc::clone(&distance),
            |a: &TropicalWeight, b: &TropicalWeight| natural_less(a, b),
            distance_estimate::<i32, TropicalWeight>(Rc::clone(&beta)),
        ));
        astar.enqueue(0);
        astar.enqueue(1);
        assert_eq!(drain(&mut astar), vec![1, 0]);
    }

    /// A trivial estimate leaves the order exactly as Dijkstra's would have it.
    #[test]
    fn a_trivial_a_star_estimate_is_dijkstra() {
        let distance = distance_of(&[5.0, 2.0, 9.0]);
        let mut queue = ShortestFirstQueue::new(a_star_compare(
            Rc::clone(&distance),
            |a: &TropicalWeight, b: &TropicalWeight| natural_less(a, b),
            trivial_estimate::<i32, TropicalWeight>(),
        ));
        for s in 0..3 {
            queue.enqueue(s);
        }
        assert_eq!(drain(&mut queue), vec![1, 0, 2]);
    }

    // --- The contract of this file: the discipline must not change the answer.

    /// Mohri's generic single-source shortest-distance, driven by whichever
    /// queue it is handed. The queue reads `distance` as it goes, which is why
    /// it is shared rather than owned.
    fn shortest_distance_with<Q: Queue<i32>>(
        fst: &StdVectorFst,
        queue: &mut Q,
        distance: &Distance,
    ) -> Vec<TropicalWeight> {
        let nstates = fst.num_states();
        {
            let mut d = distance.borrow_mut();
            d.clear();
            d.resize(nstates, TropicalWeight::zero());
        }
        let Some(start) = fst.start() else {
            return distance.borrow().clone();
        };
        let mut residual = vec![TropicalWeight::zero(); nstates];
        distance.borrow_mut()[start as usize] = TropicalWeight::one();
        residual[start as usize] = TropicalWeight::one();
        let mut enqueued = vec![false; nstates];
        enqueued[start as usize] = true;
        queue.enqueue(start);

        let mut steps = 0;
        while let Some(state) = queue.dequeue() {
            steps += 1;
            assert!(steps < 200_000, "the queue is not draining");
            enqueued[state as usize] = false;
            let r = residual[state as usize];
            residual[state as usize] = TropicalWeight::zero();
            for arc in fst.arcs(state) {
                let next = arc.nextstate() as usize;
                let contribution = r.times(arc.weight());
                let old = distance.borrow()[next];
                let new = old.plus(&contribution);
                if old == new {
                    continue;
                }
                distance.borrow_mut()[next] = new;
                residual[next] = residual[next].plus(&contribution);
                if enqueued[next] {
                    queue.update(arc.nextstate());
                } else {
                    enqueued[next] = true;
                    queue.enqueue(arc.nextstate());
                }
            }
        }
        distance.borrow().clone()
    }

    /// Bellman-Ford, as the answer every discipline has to agree with.
    fn reference_distance(fst: &StdVectorFst) -> Vec<TropicalWeight> {
        let nstates = fst.num_states();
        let mut d = vec![TropicalWeight::zero(); nstates];
        let Some(start) = fst.start() else { return d };
        d[start as usize] = TropicalWeight::one();
        for _ in 0..nstates {
            let mut changed = false;
            for s in 0..nstates {
                for arc in fst.arcs(s as i32) {
                    let relaxed = d[s].times(arc.weight());
                    let next = arc.nextstate() as usize;
                    let new = d[next].plus(&relaxed);
                    if new != d[next] {
                        d[next] = new;
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }
        d
    }

    fn random_fst(
        next: &mut impl FnMut(usize) -> usize,
        acyclic: bool,
        weighted: bool,
    ) -> StdVectorFst {
        let nstates = 2 + next(7);
        let mut fst = StdVectorFst::new();
        for _ in 0..nstates {
            fst.add_state();
        }
        fst.set_start(0);
        for s in 0..nstates {
            for _ in 0..next(4) {
                let target = if acyclic {
                    if s + 1 >= nstates {
                        continue;
                    }
                    s + 1 + next(nstates - s - 1)
                } else {
                    next(nstates)
                };
                let label = 1 + next(3) as i32;
                let weight = if weighted {
                    TropicalWeight(next(6) as f32)
                } else {
                    TropicalWeight::one()
                };
                fst.add_arc(s as i32, StdArc::new(label, label, weight, target as i32));
            }
            if next(3) == 0 {
                fst.set_final(s as i32, TropicalWeight::one());
            }
        }
        fst.properties(K_FST_PROPERTIES, true);
        fst
    }

    /// Every discipline drives the same fixed point. This is the whole reason
    /// the disciplines are interchangeable, so it is checked over random FSTs
    /// rather than over one example.
    #[test]
    fn the_discipline_never_changes_the_answer() {
        let mut rng = 0x0EEDF00Du64;
        let mut next = |bound: usize| {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((rng >> 33) as usize) % bound.max(1)
        };

        for round in 0..300 {
            let acyclic = round % 3 == 0;
            let weighted = round % 5 != 0;
            let fst = random_fst(&mut next, acyclic, weighted);
            let want = reference_distance(&fst);
            let distance: Distance = Rc::new(RefCell::new(Vec::new()));

            let mut got: Vec<(&str, Vec<TropicalWeight>)> = vec![
                (
                    "fifo",
                    shortest_distance_with(&fst, &mut FifoQueue::new(), &distance),
                ),
                (
                    "lifo",
                    shortest_distance_with(&fst, &mut LifoQueue::new(), &distance),
                ),
                (
                    "state-order",
                    shortest_distance_with(&fst, &mut StateOrderQueue::new(), &distance),
                ),
                (
                    "shortest-first",
                    shortest_distance_with(
                        &fst,
                        &mut ShortestFirstQueue::new(natural(&distance)),
                        &distance,
                    ),
                ),
                (
                    "auto",
                    shortest_distance_with(
                        &fst,
                        &mut AutoQueue::new(&fst, Some(natural(&distance))),
                        &distance,
                    ),
                ),
            ];
            if acyclic && let Some(mut top) = TopOrderQueue::new(&fst) {
                got.push((
                    "top-order",
                    shortest_distance_with(&fst, &mut top, &distance),
                ));
            }

            for (name, distances) in got {
                assert_eq!(distances, want, "round {round}, {name} discipline");
            }
        }
    }

    /// And the meta-queue in particular, on the FSTs that reach it: cyclic,
    /// weighted, and with more than one component.
    #[test]
    fn the_meta_queue_settles_the_same_distances() {
        let mut rng = 0x5CC_0EEDu64;
        let mut next = |bound: usize| {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((rng >> 33) as usize) % bound.max(1)
        };

        let mut reached = 0;
        for round in 0..200 {
            let fst = random_fst(&mut next, false, true);
            let distance: Distance = Rc::new(RefCell::new(Vec::new()));
            let mut auto = AutoQueue::new(&fst, Some(natural(&distance)));
            if auto.chosen() != QueueType::Scc {
                continue;
            }
            reached += 1;
            let got = shortest_distance_with(&fst, &mut auto, &distance);
            assert_eq!(got, reference_distance(&fst), "round {round}");
        }
        assert!(reached > 20, "only {reached} FSTs reached the meta-queue");
    }
}
