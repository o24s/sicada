//! Functions and classes to create a partition of states.
//!
//! Defines a partitioning of elements, used to represent equivalence classes
//! for FST operations like minimization.
//!
//! The elements are numbered from `0` to `num_elements - 1`.
//!
//! We maintain a partition of these elements into classes. The classes are also
//! numbered from zero. We also support a specialized interface that allows
//! efficiently splitting classes in the Hopcroft minimization algorithm.
//!
//! The split interface maintains a binary partition of every class into a 'yes'
//! and a 'no' subset; every element starts in 'no'. [`Partition::split_on`]
//! moves one element to the 'yes' subset of its class and records the class as
//! visited. [`Partition::finalize_split`] then, for each visited class whose two
//! subsets are both non-empty, splits off the *smaller* subset as a new class and
//! leaves the larger one in place, which is the bound that keeps Hopcroft's
//! algorithm at `O(E log V)`, before resetting everything back into the 'no'
//! subsets.
//!
//! Port of OpenFst's `partition.h`.

const K_NULL_ID: usize = usize::MAX;

/// Information about a given element.
#[derive(Debug, Clone, Copy)]
struct Element {
    /// Class ID of this element.
    class_id: usize,
    /// Interpreted as a bool: `yes == yes_counter` means it's in the 'yes' set.
    yes: usize,
    /// Next element in the 'no' list or 'yes' list of this class.
    next_element: usize,
    /// Previous element in the 'no' or 'yes' doubly linked list.
    prev_element: usize,
}

impl Default for Element {
    #[inline(always)]
    fn default() -> Self {
        Self {
            class_id: K_NULL_ID,
            yes: 0,
            next_element: K_NULL_ID,
            prev_element: K_NULL_ID,
        }
    }
}

/// Information about a given class.
#[derive(Debug, Clone, Copy)]
struct Class {
    /// Total number of elements in this class ('no' plus 'yes' subsets).
    size: usize,
    /// Total number of elements of 'yes' subset of this class.
    yes_size: usize,
    /// Index of head element of doubly-linked list in 'no' subset.
    no_head: usize,
    /// Index of head element of doubly-linked list in 'yes' subset.
    yes_head: usize,
}

impl Default for Class {
    #[inline(always)]
    fn default() -> Self {
        Self {
            size: 0,
            yes_size: 0,
            no_head: K_NULL_ID,
            yes_head: K_NULL_ID,
        }
    }
}

/// Defines a partitioning of elements into equivalence classes.
#[derive(Debug, Clone)]
pub struct Partition {
    elements: Vec<Element>,
    classes: Vec<Class>,
    visited_classes: Vec<usize>,
    yes_counter: usize,
}

impl Default for Partition {
    fn default() -> Self {
        Self::new(0)
    }
}

impl Partition {
    /// Creates an empty partition for `num_elements`.
    pub fn new(num_elements: usize) -> Self {
        let mut p = Self {
            elements: Vec::new(),
            classes: Vec::new(),
            visited_classes: Vec::new(),
            yes_counter: 1,
        };
        p.initialize(num_elements);
        p
    }

    /// Initializes or resets the partition for `num_elements`.
    /// Elements are initially unassigned.
    pub fn initialize(&mut self, num_elements: usize) {
        self.elements.clear();
        self.elements.resize(num_elements, Element::default());
        self.classes.clear();
        self.classes.reserve(num_elements);
        // SICADA-DIVERGE: upstream's Initialize leaves visited_classes_ alone, so
        // reinitializing a partition that had a pending SplitOn carries stale
        // class ids into the next FinalizeSplit, which then indexes a classes_
        // vector that has just been cleared. Costs nothing to reset here.
        self.visited_classes.clear();
        self.yes_counter = 1;
    }

    /// Adds a class; returns new number of classes.
    #[inline]
    pub fn add_class(&mut self) -> usize {
        let num_classes = self.classes.len();
        self.classes.push(Class::default());
        num_classes
    }

    /// Adds `num_classes` new (empty) classes.
    #[inline]
    pub fn allocate_classes(&mut self, num_classes: usize) {
        self.classes
            .resize(self.classes.len() + num_classes, Class::default());
    }

    /// Adds `element_id` to `class_id`.
    /// `element_id` must not currently be a member of any class.
    #[inline]
    pub fn add(&mut self, element_id: usize, class_id: usize) {
        self.classes[class_id].size += 1;
        let no_head = self.classes[class_id].no_head;

        if no_head != K_NULL_ID {
            self.elements[no_head].prev_element = element_id;
        }
        self.classes[class_id].no_head = element_id;

        let element = &mut self.elements[element_id];
        element.class_id = class_id;
        element.yes = 0; // Added to the 'no' subset.
        element.next_element = no_head;
        element.prev_element = K_NULL_ID;
    }

    /// Moves `element_id` from the 'no' subset of its current class to the 'no' subset
    /// of `class_id`.
    ///
    /// Must not be called between [`split_on`](Self::split_on) and
    /// [`finalize_split`](Self::finalize_split): the element is excised from
    /// whichever list it is in on the assumption that it is the 'no' list.
    #[inline]
    pub fn move_element(&mut self, element_id: usize, class_id: usize) {
        debug_assert!(
            self.visited_classes.is_empty(),
            "move_element between split_on and finalize_split corrupts the class lists"
        );
        let old_class_id = self.elements[element_id].class_id;
        self.classes[old_class_id].size -= 1;

        let prev = self.elements[element_id].prev_element;
        let next = self.elements[element_id].next_element;

        if prev != K_NULL_ID {
            self.elements[prev].next_element = next;
        } else {
            self.classes[old_class_id].no_head = next;
        }

        if next != K_NULL_ID {
            self.elements[next].prev_element = prev;
        }

        self.add(element_id, class_id);
    }

    /// Moves `element_id` to the 'yes' subset of its class if it was in the 'no'
    /// subset, and marks the class as having been visited.
    #[inline]
    pub fn split_on(&mut self, element_id: usize) {
        if self.elements[element_id].yes == self.yes_counter {
            return; // Already in the 'yes' set.
        }

        let class_id = self.elements[element_id].class_id;
        let prev = self.elements[element_id].prev_element;
        let next = self.elements[element_id].next_element;

        // Excise from 'no' list
        if prev != K_NULL_ID {
            self.elements[prev].next_element = next;
        } else {
            self.classes[class_id].no_head = next;
        }

        if next != K_NULL_ID {
            self.elements[next].prev_element = prev;
        }

        // Add to 'yes' list
        let yes_head = self.classes[class_id].yes_head;
        if yes_head != K_NULL_ID {
            self.elements[yes_head].prev_element = element_id;
        } else {
            self.visited_classes.push(class_id);
        }

        let element = &mut self.elements[element_id];
        element.yes = self.yes_counter;
        element.next_element = yes_head;
        element.prev_element = K_NULL_ID;

        self.classes[class_id].yes_head = element_id;
        self.classes[class_id].yes_size += 1;
    }

    /// Creates a new class containing the smaller of the two subsets of elements
    /// for each class that has a nontrivial split. The provided closure `enqueue`
    /// is called with the identifier of the newly created classes.
    #[inline]
    pub fn finalize_split<F>(&mut self, mut enqueue: F)
    where
        F: FnMut(usize),
    {
        for i in 0..self.visited_classes.len() {
            let visited_class = self.visited_classes[i];
            if let Some(new_class) = self.split_refine(visited_class) {
                enqueue(new_class);
            }
        }
        self.visited_classes.clear();
        self.yes_counter += 1; // Sets all 'yes' members to false implicitly.
    }

    #[inline(always)]
    pub fn class_id(&self, element_id: usize) -> usize {
        self.elements[element_id].class_id
    }

    #[inline(always)]
    pub fn class_size(&self, class_id: usize) -> usize {
        self.classes[class_id].size
    }

    #[inline(always)]
    pub fn num_classes(&self) -> usize {
        self.classes.len()
    }

    /// Returns an iterator over the elements in the 'no' subset of a class.
    /// (After `finalize_split`, all elements are in the 'no' subset).
    #[inline]
    pub fn iter_class(&self, class_id: usize) -> PartitionClassIter<'_> {
        PartitionClassIter {
            partition: self,
            current_id: self.classes[class_id].no_head,
        }
    }

    #[inline]
    fn split_refine(&mut self, class_id: usize) -> Option<usize> {
        let yes_size = self.classes[class_id].yes_size;
        let size = self.classes[class_id].size;
        let no_size = size - yes_size;

        if no_size == 0 {
            self.classes[class_id].no_head = self.classes[class_id].yes_head;
            self.classes[class_id].yes_head = K_NULL_ID;
            self.classes[class_id].yes_size = 0;
            None
        } else {
            let new_class_id = self.classes.len();
            self.classes.push(Class::default());

            if no_size < yes_size {
                let old_no_head = self.classes[class_id].no_head;
                let old_yes_head = self.classes[class_id].yes_head;

                // Move 'no' subset to new class
                self.classes[new_class_id].no_head = old_no_head;
                self.classes[new_class_id].size = no_size;

                // Move 'yes' subset to old class ('no' subset)
                self.classes[class_id].no_head = old_yes_head;
                self.classes[class_id].yes_head = K_NULL_ID;
                self.classes[class_id].size = yes_size;
                self.classes[class_id].yes_size = 0;
            } else {
                let old_yes_head = self.classes[class_id].yes_head;

                // Move 'yes' subset to new class
                self.classes[new_class_id].size = yes_size;
                self.classes[new_class_id].no_head = old_yes_head;

                // Retain only 'no' subset in old class
                self.classes[class_id].size = no_size;
                self.classes[class_id].yes_size = 0;
                self.classes[class_id].yes_head = K_NULL_ID;
            }

            // Update class_id mapping for all moved elements
            let mut e = self.classes[new_class_id].no_head;
            while e != K_NULL_ID {
                self.elements[e].class_id = new_class_id;
                e = self.elements[e].next_element;
            }

            Some(new_class_id)
        }
    }
}

/// An iterator over the elements in the 'no' subset of a class.
///
/// `Clone` stands in for upstream's `PartitionIterator::Reset`.
#[derive(Clone)]
pub struct PartitionClassIter<'a> {
    partition: &'a Partition,
    current_id: usize,
}

impl<'a> Iterator for PartitionClassIter<'a> {
    type Item = usize;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        if self.current_id == K_NULL_ID {
            None
        } else {
            let val = self.current_id;
            self.current_id = self.partition.elements[val].next_element;
            Some(val)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    impl Partition {
        /// Collects the members of a class, sorted, for comparison in tests.
        fn class_members(&self, class_id: usize) -> Vec<usize> {
            let mut members: Vec<_> = self.iter_class(class_id).collect();
            members.sort_unstable();
            members
        }

        /// Checks the structural invariants that every operation must preserve:
        /// each class list is consistent with the recorded sizes, every element
        /// appears in exactly the class it names, and the links are coherent in
        /// both directions.
        fn assert_consistent(&self) {
            let mut seen = vec![false; self.elements.len()];
            for class_id in 0..self.num_classes() {
                let mut count = 0;
                let mut prev = K_NULL_ID;
                let mut e = self.classes[class_id].no_head;
                while e != K_NULL_ID {
                    assert!(!seen[e], "element {e} appears in more than one class list");
                    seen[e] = true;
                    assert_eq!(
                        self.elements[e].class_id, class_id,
                        "element {e} is linked into class {class_id} but names another"
                    );
                    assert_eq!(
                        self.elements[e].prev_element, prev,
                        "back link broken at element {e}"
                    );
                    prev = e;
                    e = self.elements[e].next_element;
                    count += 1;
                }
                // Outside a split, everything lives in the 'no' subset.
                if self.visited_classes.is_empty() {
                    assert_eq!(
                        count,
                        self.class_size(class_id),
                        "class {class_id} size disagrees with its list"
                    );
                    assert_eq!(self.classes[class_id].yes_head, K_NULL_ID);
                    assert_eq!(self.classes[class_id].yes_size, 0);
                }
            }
        }
    }

    #[test]
    fn test_partition_basic() {
        let mut p = Partition::new(5);
        p.allocate_classes(2);

        // Class 0: {0, 1}
        p.add(0, 0);
        p.add(1, 0);

        // Class 1: {2, 3, 4}
        p.add(2, 1);
        p.add(3, 1);
        p.add(4, 1);

        assert_eq!(p.num_classes(), 2);
        assert_eq!(p.class_size(0), 2);
        assert_eq!(p.class_size(1), 3);

        // Elements iter
        let c0_elements: Vec<_> = p.iter_class(0).collect();
        assert_eq!(c0_elements.len(), 2);
        assert!(c0_elements.contains(&0) && c0_elements.contains(&1));

        // Split class 1 on element 2 -> 'yes'={2}, 'no'={3,4}
        p.split_on(2);

        let mut enqueued = Vec::new();
        p.finalize_split(|new_class| enqueued.push(new_class));

        // Since 'yes_size' (1) < 'no_size' (2), the new class gets the smaller subset {2}.
        assert_eq!(enqueued.len(), 1);
        assert_eq!(p.num_classes(), 3);

        // Verify class sizes after split
        assert_eq!(p.class_size(0), 2);

        let new_class_id = enqueued[0];

        // Element 2 should now be in the newly created class
        assert_eq!(p.class_id(2), new_class_id);
        assert_eq!(p.class_size(new_class_id), 1);

        // Elements 3 and 4 should remain in class 1
        assert_eq!(p.class_id(3), 1);
        assert_eq!(p.class_id(4), 1);
        assert_eq!(p.class_size(1), 2);
    }

    #[test]
    fn test_partition_move() {
        let mut p = Partition::new(3);
        p.allocate_classes(2);

        p.add(0, 0);
        p.add(1, 0);
        p.add(2, 1);

        assert_eq!(p.class_size(0), 2);
        assert_eq!(p.class_size(1), 1);

        // Move element 1 from Class 0 to Class 1
        p.move_element(1, 1);

        assert_eq!(p.class_size(0), 1);
        assert_eq!(p.class_size(1), 2);
        assert_eq!(p.class_id(1), 1);
    }
    /// The larger subset must stay in the original class and the smaller one
    /// become the new class. That asymmetry is what bounds Hopcroft's algorithm,
    /// so it is worth pinning in both directions.
    #[test]
    fn finalize_split_moves_out_the_smaller_subset() {
        // 'yes' smaller than 'no': {0} splits out of {0,1,2,3}.
        let mut p = Partition::new(4);
        p.allocate_classes(1);
        for element in 0..4 {
            p.add(element, 0);
        }
        p.split_on(0);
        let mut created = Vec::new();
        p.finalize_split(|class_id| created.push(class_id));
        assert_eq!(created.len(), 1);
        assert_eq!(p.class_members(created[0]), vec![0]);
        assert_eq!(p.class_members(0), vec![1, 2, 3]);
        p.assert_consistent();

        // 'yes' larger than 'no': {0,1,2} splits out of {0,1,2,3}, so the new
        // class receives the single 'no' element instead.
        let mut p = Partition::new(4);
        p.allocate_classes(1);
        for element in 0..4 {
            p.add(element, 0);
        }
        for element in 0..3 {
            p.split_on(element);
        }
        let mut created = Vec::new();
        p.finalize_split(|class_id| created.push(class_id));
        assert_eq!(created.len(), 1);
        assert_eq!(p.class_members(created[0]), vec![3]);
        assert_eq!(p.class_members(0), vec![0, 1, 2]);
        p.assert_consistent();
    }

    #[test]
    fn splitting_every_member_creates_no_new_class() {
        let mut p = Partition::new(3);
        p.allocate_classes(1);
        for element in 0..3 {
            p.add(element, 0);
        }
        for element in 0..3 {
            p.split_on(element);
        }
        let mut created = Vec::new();
        p.finalize_split(|class_id| created.push(class_id));
        assert!(created.is_empty());
        assert_eq!(p.num_classes(), 1);
        assert_eq!(p.class_members(0), vec![0, 1, 2]);
        p.assert_consistent();
    }

    #[test]
    fn split_on_is_idempotent_within_a_round() {
        let mut p = Partition::new(3);
        p.allocate_classes(1);
        for element in 0..3 {
            p.add(element, 0);
        }
        p.split_on(1);
        p.split_on(1);
        p.split_on(1);
        let mut created = Vec::new();
        p.finalize_split(|class_id| created.push(class_id));
        assert_eq!(created.len(), 1);
        assert_eq!(p.class_members(created[0]), vec![1]);
        assert_eq!(p.class_members(0), vec![0, 2]);
        p.assert_consistent();
    }

    #[test]
    fn several_classes_split_in_one_round() {
        let mut p = Partition::new(6);
        p.allocate_classes(2);
        for element in 0..3 {
            p.add(element, 0);
        }
        for element in 3..6 {
            p.add(element, 1);
        }
        p.split_on(0);
        p.split_on(4);
        let mut created = Vec::new();
        p.finalize_split(|class_id| created.push(class_id));
        assert_eq!(created.len(), 2);
        assert_eq!(p.num_classes(), 4);
        assert_eq!(p.class_members(0), vec![1, 2]);
        assert_eq!(p.class_members(1), vec![3, 5]);
        assert_eq!(p.class_id(0), created[0]);
        assert_eq!(p.class_id(4), created[1]);
        p.assert_consistent();
    }

    /// `yes` membership is expressed as `element.yes == yes_counter`, so the
    /// counter bump in `finalize_split` is what clears the flags. Successive
    /// rounds must not see the previous round's marks.
    #[test]
    fn successive_rounds_start_with_an_empty_yes_subset() {
        let mut p = Partition::new(8);
        p.allocate_classes(1);
        for element in 0..8 {
            p.add(element, 0);
        }

        let mut sizes = Vec::new();
        for round in 0..3 {
            p.split_on(round);
            let mut created = Vec::new();
            p.finalize_split(|class_id| created.push(class_id));
            assert_eq!(created.len(), 1, "round {round}");
            assert_eq!(p.class_members(created[0]), vec![round], "round {round}");
            sizes.push(p.class_size(0));
            p.assert_consistent();
        }
        assert_eq!(sizes, vec![7, 6, 5]);
    }

    #[test]
    fn initialize_resets_a_partition_with_a_pending_split() {
        // Upstream leaves visited_classes_ populated here, so the next
        // FinalizeSplit would index the freshly cleared classes_ vector.
        let mut p = Partition::new(4);
        p.allocate_classes(2);
        for element in 0..4 {
            p.add(element, element % 2);
        }
        p.split_on(0);

        p.initialize(2);
        p.allocate_classes(1);
        p.add(0, 0);
        p.add(1, 0);
        let mut created = Vec::new();
        p.finalize_split(|class_id| created.push(class_id));
        assert!(created.is_empty());
        assert_eq!(p.num_classes(), 1);
        p.assert_consistent();
    }

    #[test]
    fn iterating_a_class_is_repeatable() {
        let mut p = Partition::new(3);
        p.allocate_classes(1);
        for element in 0..3 {
            p.add(element, 0);
        }
        let iter = p.iter_class(0);
        let first: Vec<_> = iter.clone().collect();
        let second: Vec<_> = iter.collect();
        assert_eq!(first, second);
    }

    /// Randomized refinement: repeatedly split on an arbitrary subset and check
    /// the invariants hold and that classes only ever get finer.
    #[test]
    fn random_refinement_keeps_the_partition_consistent() {
        let mut state = 0xDEAD_BEEF_1234_5678u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };

        const ELEMENTS: usize = 64;
        let mut p = Partition::new(ELEMENTS);
        p.allocate_classes(1);
        for element in 0..ELEMENTS {
            p.add(element, 0);
        }

        let mut previous_classes = 1;
        for _ in 0..200 {
            for element in 0..ELEMENTS {
                if next() % 2 == 0 {
                    p.split_on(element);
                }
            }
            let mut created = Vec::new();
            p.finalize_split(|class_id| created.push(class_id));
            p.assert_consistent();

            assert!(
                p.num_classes() >= previous_classes,
                "refinement went backwards"
            );
            previous_classes = p.num_classes();

            // Every element belongs to exactly one class, and the class sizes
            // account for all of them.
            let total: usize = (0..p.num_classes()).map(|c| p.class_size(c)).sum();
            assert_eq!(total, ELEMENTS);
            for element in 0..ELEMENTS {
                let class_id = p.class_id(element);
                assert!(p.iter_class(class_id).any(|member| member == element));
            }
        }
    }
}
