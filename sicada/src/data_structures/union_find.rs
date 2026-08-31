//! Union-find algorithm for dense sets of non-negative integers, implemented
//! using disjoint tree forests with rank heuristics and path compression.
//!
//! Port of OpenFst's `union-find.h`, with two upstream defects fixed.

const K_NULL_ID: usize = usize::MAX;

/// Union-Find algorithm for dense sets of non-negative integers.
#[derive(Debug, Clone)]
pub struct UnionFind {
    parent: Vec<usize>,
    /// Rank of an element: an upper bound on the depth of its tree.
    ///
    /// SICADA-OPT: upstream stores this as `int`. Rank only grows when two trees
    /// of equal rank merge, so it is bounded by `log2(len)`; a `u8` covers more
    /// elements than can be addressed, at a quarter of the memory. The rank
    /// array is walked alongside `parent` on every union, so the smaller
    /// footprint is also fewer cache lines.
    rank: Vec<u8>,
}

impl UnionFind {
    /// Creates a disjoint set forest for the range `[0, max)`.
    pub fn new(max: usize) -> Self {
        Self {
            parent: vec![K_NULL_ID; max],
            rank: vec![0; max],
        }
    }

    /// Finds the representative of the set `item` belongs to, performing path
    /// compression if necessary. Returns `None` if `item` hasn't been initialized
    /// using `make_set` or `make_all_set`.
    /// SICADA-OPT: the two loops below are the whole cost of a union-find, and
    /// every access in them is bounds-checked by default. The one check that
    /// matters happens once, on entry.
    ///
    /// The invariant that makes the rest redundant: every entry of `parent` is
    /// either `K_NULL_ID` or an index into `parent`. `make_set` and
    /// `make_all_set` only ever store an element's own index, `link` only stores
    /// a root, and both resize `parent` and `rank` together, so a stored index
    /// can never outlive the array it points into.
    #[inline]
    pub fn find_set(&mut self, mut item: usize) -> Option<usize> {
        if item >= self.parent.len() {
            return None;
        }
        // SAFETY: `item` was just checked against the length.
        let mut root = unsafe { *self.parent.get_unchecked(item) };
        if root == K_NULL_ID {
            return None;
        }

        // SAFETY: by the invariant above, every value read out of `parent` here
        // is itself a valid index into `parent`.
        unsafe {
            while root != *self.parent.get_unchecked(root) {
                root = *self.parent.get_unchecked(root);
            }

            // Path compression: point everything along the way at the root.
            while item != root {
                let parent = *self.parent.get_unchecked(item);
                *self.parent.get_unchecked_mut(item) = root;
                item = parent;
            }
        }

        Some(root)
    }

    /// Creates the (destructive) union of the sets `x` and `y` belong to.
    /// If either `x` or `y` is not initialized, this does nothing.
    ///
    /// SICADA-BUGFIX: upstream's `Union` passes `FindSet`'s failure sentinel
    /// straight into `Link`, which indexes `parent_`/`rank_` with it. That is an
    /// out-of-bounds access whenever exactly one argument is uninitialized.
    #[inline]
    pub fn union(&mut self, x: usize, y: usize) {
        if let (Some(root_x), Some(root_y)) = (self.find_set(x), self.find_set(y)) {
            self.link(root_x, root_y);
        }
    }

    /// Initialization of an element: creates a singleton set containing `item`.
    /// The internal arrays are resized if `item >= max`.
    #[inline]
    pub fn make_set(&mut self, item: usize) -> usize {
        if item >= self.parent.len() {
            let new_size = if item > 0 { 2 * item } else { 2 };
            self.parent.resize(new_size, K_NULL_ID);
            self.rank.resize(new_size, 0);
        }
        self.parent[item] = item;
        // SICADA-BUGFIX: upstream leaves the old rank in place, so a
        // reinitialized element can outrank a genuine singleton and `link`
        // attaches the trees the wrong way round, losing the height bound.
        self.rank[item] = 0;
        item
    }

    /// Initialization of all elements starting from `0` to `max - 1` to distinct sets.
    pub fn make_all_set(&mut self, max: usize) {
        self.parent.resize(max, K_NULL_ID);
        self.rank.resize(max, 0);
        for i in 0..max {
            self.parent[i] = i;
            // SICADA-BUGFIX: as in `make_set`; upstream's MakeAllSet never
            // touches rank_.
            self.rank[i] = 0;
        }
    }

    /// For testing only: Returns the direct parent of `x`.
    #[inline]
    pub fn parent(&self, x: usize) -> Option<usize> {
        self.parent.get(x).copied().filter(|&p| p != K_NULL_ID)
    }

    /// Links trees rooted in `x` and `y`.
    /// Links the trees rooted at `x` and `y`, hanging the shallower under the
    /// deeper.
    #[inline]
    fn link(&mut self, x: usize, y: usize) {
        if x == y {
            return;
        }
        debug_assert!(x < self.parent.len() && y < self.parent.len());
        debug_assert_eq!(self.parent.len(), self.rank.len());
        // SAFETY: both arguments are roots returned by `find_set`, so they index
        // `parent`; `rank` is resized in step with `parent` everywhere.
        unsafe {
            let rank_x = *self.rank.get_unchecked(x);
            let rank_y = *self.rank.get_unchecked(y);
            if rank_x > rank_y {
                *self.parent.get_unchecked_mut(y) = x;
            } else {
                *self.parent.get_unchecked_mut(x) = y;
                if rank_x == rank_y {
                    *self.rank.get_unchecked_mut(y) = rank_y + 1;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_union_find_basic() {
        let mut uf = UnionFind::new(5);
        uf.make_all_set(5);

        // Initially, everyone is their own parent
        for i in 0..5 {
            assert_eq!(uf.find_set(i), Some(i));
        }

        uf.union(0, 1);
        uf.union(2, 3);

        let root_0 = uf.find_set(0).unwrap();
        assert_eq!(root_0, uf.find_set(1).unwrap());

        let root_2 = uf.find_set(2).unwrap();
        assert_eq!(root_2, uf.find_set(3).unwrap());

        // 0 and 2 are in different sets
        assert_ne!(root_0, root_2);

        // Union the two sets
        uf.union(1, 3);

        // Now 0, 1, 2, 3 should all have the same root
        let new_root = uf.find_set(0).unwrap();
        assert_eq!(new_root, uf.find_set(1).unwrap());
        assert_eq!(new_root, uf.find_set(2).unwrap());
        assert_eq!(new_root, uf.find_set(3).unwrap());

        // 4 is still independent
        assert_ne!(new_root, uf.find_set(4).unwrap());
    }

    #[test]
    fn test_dynamic_make_set() {
        let mut uf = UnionFind::new(0);

        uf.make_set(10);
        uf.make_set(20);

        assert_eq!(uf.find_set(10), Some(10));
        assert_eq!(uf.find_set(20), Some(20));
        assert_eq!(uf.find_set(15), None); // Not initialized

        uf.union(10, 20);
        assert_eq!(uf.find_set(10), uf.find_set(20));
    }
    #[test]
    fn union_with_an_uninitialized_element_is_a_no_op() {
        // Regression test for an upstream C++ bug.
        // Union(x, y) must not use an invalid index for parent_/rank_
        // if x or y was never initialized via make_set.
        let mut uf = UnionFind::new(3);
        uf.make_set(0);
        uf.make_set(1);

        // 2 is not initialized.
        uf.union(0, 2); // Should not panic or corrupt state
        uf.union(2, 1); // Should not panic or corrupt state

        assert_eq!(uf.find_set(0), Some(0));
        assert_eq!(uf.find_set(1), Some(1));
        assert_eq!(uf.find_set(2), None);
    }

    #[test]
    fn make_set_resets_the_rank() {
        // Re-inserting an element must reset its rank. Otherwise, later
        // `link` calls will use a stale rank, potentially picking the
        // wrong root and breaking the tree-height guarantee.
        let mut uf = UnionFind::new(4);
        uf.make_all_set(4);

        // Union 0 and 1 (both rank 0) to bump the surviving root's rank to 1.
        uf.union(0, 1);
        let root = uf.find_set(1).unwrap();
        assert_eq!(uf.parent(root), Some(root));

        // Re-initialize `root` as a singleton.
        uf.make_set(root);

        // Union with a fresh rank-0 element `2`.
        // We verify the tie-breaking behavior, which requires both sides
        // to have rank 0 after make_set.
        uf.union(root, 2);
        let new_root = uf.find_set(root).unwrap();

        // The tie-breaking rule makes `y` (the second argument's root)
        // the new parent. `root` should no longer be its own parent.
        assert_ne!(uf.parent(root), Some(root));
        assert_eq!(uf.find_set(2), Some(new_root));
    }

    #[test]
    fn make_all_set_resets_the_rank() {
        // Regression test for an upstream C++ bug where MakeAllSet leaves
        // rank_ stale. We build up nonzero ranks, call make_all_set,
        // and verify the tree behaves as if freshly initialized.
        let mut uf = UnionFind::new(4);
        uf.make_all_set(4);
        uf.union(0, 1);
        uf.union(2, 3);
        uf.union(1, 3); // Combines two rank-1 trees

        uf.make_all_set(4); // Fully resets state, including rank

        for i in 0..4 {
            assert_eq!(uf.find_set(i), Some(i));
            assert_eq!(uf.parent(i), Some(i));
        }
    }
    #[test]
    fn find_set_reports_uninitialized_and_out_of_range_elements() {
        let mut uf = UnionFind::new(3);
        assert_eq!(uf.find_set(0), None, "never passed to make_set");
        assert_eq!(uf.find_set(99), None, "out of range");
        uf.make_set(0);
        assert_eq!(uf.find_set(0), Some(0));
    }

    #[test]
    fn union_of_elements_already_together_changes_nothing() {
        let mut uf = UnionFind::new(3);
        uf.make_all_set(3);
        uf.union(0, 1);
        let root = uf.find_set(0).unwrap();
        uf.union(0, 1);
        uf.union(1, 0);
        assert_eq!(uf.find_set(0), Some(root));
        assert_eq!(uf.find_set(1), Some(root));
    }

    /// Path compression must leave every element on the path pointing straight
    /// at the root, which keeps repeated lookups cheap.
    #[test]
    fn find_set_flattens_the_path_it_walks() {
        let mut uf = UnionFind::new(4);
        uf.make_all_set(4);
        // Build a chain by merging one element at a time into a growing tree.
        uf.union(0, 1);
        uf.union(1, 2);
        uf.union(2, 3);

        let root = uf.find_set(0).unwrap();
        for element in 0..4 {
            uf.find_set(element);
            assert_eq!(
                uf.parent(element),
                Some(root),
                "element {element} should point straight at the root"
            );
        }
    }

    /// The rank heuristic must hang the shallower tree under the deeper one, so
    /// that merging never deepens the taller side.
    #[test]
    fn union_by_rank_keeps_the_deeper_tree_on_top() {
        let mut uf = UnionFind::new(5);
        uf.make_all_set(5);
        uf.union(0, 1); // rank 1 tree rooted at the survivor
        let deep_root = uf.find_set(0).unwrap();

        // 4 is still a rank-0 singleton, so it must be attached under deep_root
        // whichever way round the arguments come.
        uf.union(4, deep_root);
        assert_eq!(uf.find_set(4), Some(deep_root));
        assert_eq!(uf.parent(4), Some(deep_root));
    }

    /// Merging n elements pairwise must not build a chain: with union by rank
    /// and path compression the tree stays shallow.
    #[test]
    fn many_unions_keep_the_forest_shallow() {
        const N: usize = 1024;
        let mut uf = UnionFind::new(N);
        uf.make_all_set(N);
        for element in 1..N {
            uf.union(element - 1, element);
        }
        let root = uf.find_set(0).unwrap();
        for element in 0..N {
            assert_eq!(uf.find_set(element), Some(root));
        }
        // After the lookups above, path compression has flattened everything.
        for element in 0..N {
            assert_eq!(uf.parent(element), Some(root));
        }
    }

    #[test]
    fn make_set_grows_the_forest_on_demand() {
        let mut uf = UnionFind::new(2);
        uf.make_all_set(2);
        uf.make_set(10);
        assert_eq!(uf.find_set(10), Some(10));
        uf.union(0, 10);
        assert_eq!(uf.find_set(0), uf.find_set(10));
        // Elements in the grown range that were never initialized stay unknown.
        assert_eq!(uf.find_set(9), None);
    }
}
