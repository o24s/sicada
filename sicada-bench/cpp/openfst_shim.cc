// Copyright 2026 The OpenFst Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Benchmark shim exposing the OpenFst data structures sicada has ported.
//
// `Heap`, `UnionFind`, `ArcArena` and `CompactSet` are copied from OpenFst, out
// of heap.h, union-find.h, arc-arena.h and util.h under
// vendor/openfst/openfst/lib, so that the comparison runs against upstream's
// real code; they carry its copyright above. They are changed in one way: the
// absl logging macros are dropped, since they are not part of what is being
// measured, and keeping the classes here rather than including the headers
// avoids pulling in absl, which is not vendored. Everything from `BenchArc`
// down is this benchmark's own.
//
// Everything is compiled at the same optimisation level as the Rust side and
// driven through the same operation sequence, so the two are doing equal work.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <list>
#include <memory>
#include <set>
#include <utility>
#include <vector>

namespace fst {

// A templated heap implementation that supports in-place update of values.
//
// The templated heap implementation is a little different from the STL
// priority_queue and the *_heap operations in STL. This heap supports
// indexing of values in the heap via an associated key.
//
// Each value is internally associated with a key which is returned to the
// calling functions on heap insert. This key can be used to later update
// the specific value in the heap.
//
// T: the element type of the hash. It can be POD, Data or a pointer to Data.
// Compare: comparison functor for determining min-heapness.
template <class T, class Compare>
class Heap {
 public:
  using Value = T;

  static constexpr int kNoKey = -1;

  // Initializes with a specific comparator.
  explicit Heap(Compare comp = Compare()) : comp_(comp), size_(0) {}

  // Inserts a value into the heap.
  int Insert(const Value& value) {
    if (size_ < elements_.size()) {
      elements_[size_].value = value;
      pos_[elements_[size_].key] = size_;
    } else {
      elements_.push_back({value, size_});
      pos_.push_back(size_);
    }
    ++size_;
    return Insert(value, size_ - 1);
  }

  // Updates a value at position given by the key. The pos_ array is first
  // indexed by the key. The position gives the position in the heap array.
  // Once we have the position we can then use the standard heap operations
  // to calculate the parent and child positions.
  void Update(int key, const Value& value) {
    const auto i = pos_[key];
    const bool is_better = comp_(value, elements_[Parent(i)].value);
    elements_[i].value = value;
    if (is_better) {
      Insert(value, i);
    } else {
      Heapify(i);
    }
  }

  // Returns the least value.
  Value Pop() {
    
    const Value top = elements_.front().value;
    Swap(0, size_ - 1);
    size_--;
    Heapify(0);
    return top;
  }

  // Returns the least value w.r.t. the comparison function from the
  // heap.
  const Value& Top() const {
    
    return elements_.front().value;
  }

  // Returns the element for the given key.
  const Value& Get(int key) const {
    
    return elements_[pos_[key]].value;
  }

  // Checks if the heap is empty.
  bool Empty() const { return size_ == 0; }

  void Clear() { size_ = 0; }

  int Size() const { return size_; }

  void Reserve(int size) {
    elements_.reserve(size);
    pos_.reserve(size);
  }

  const Compare& GetCompare() const { return comp_; }

 private:
  // The following private routines are used in a supportive role
  // for managing the heap and keeping the heap properties.

  // Computes left child of parent.
  static int Left(int i) {
    return 2 * (i + 1) - 1;  // 0 -> 1, 1 -> 3
  }

  // Computes right child of parent.
  static int Right(int i) {
    return 2 * (i + 1);  // 0 -> 2, 1 -> 4
  }

  // Given a child computes parent.
  static int Parent(int i) {
    return (i - 1) / 2;  // 0 -> 0, 1 -> 0, 2 -> 0,  3 -> 1,  4 -> 1, ...
  }

  // Swaps a child and parent. Use to move element up/down tree. Note the use of
  // a little trick here. When we swap we need to swap:
  //
  // - the value
  // - the associated keys
  // - the position of the value in the heap
  void Swap(int j, int k) {
    if (j == k) return;
    pos_[elements_[j].key] = k;
    pos_[elements_[k].key] = j;
    std::swap(elements_[j], elements_[k]);
  }

  // Heapifies the subtree rooted at index i.
  void Heapify(int i) {
    while (true) {
      const auto l = Left(i);
      const auto r = Right(i);
      auto largest =
          (l < size_ && comp_(elements_[l].value, elements_[i].value)) ? l : i;
      if (r < size_ && comp_(elements_[r].value, elements_[largest].value)) {
        largest = r;
      }
      if (largest != i) {
        Swap(i, largest);
        i = largest;
      } else {
        break;
      }
    }
  }

  // Inserts (updates) element at subtree rooted at index i.
  int Insert(const Value& value, int i) {
    int p;
    while (i > 0 && !comp_(elements_[p = Parent(i)].value, value)) {
      Swap(i, p);
      i = p;
    }
    return elements_[i].key;
  }

 private:
  struct Node {
    Value value;
    int key;
  };

  const Compare comp_;

  std::vector<int> pos_;
  std::vector<Node> elements_;
  int size_;
};


// Union-Find algorithm for dense sets of non-negative integers.
template <class T>
class UnionFind {
 public:
  // Creates a disjoint set forest for the range [0; max); 'fail' is a value
  // indicating that an element hasn't been initialized using MakeSet(...).
  // The upper bound of the range can be reset (increased) using MakeSet(...).
  UnionFind(T max, T fail) : parent_(max, fail), rank_(max), fail_(fail) {}

  // Finds the representative of the set 'item' belongs to, performing path
  // compression if necessary.
  T FindSet(T item) {
    if (item >= parent_.size() || item == fail_ || parent_[item] == fail_) {
      return fail_;
    }
    T root = item;
    while (root != parent_[root]) {
      root = parent_[root];
    }
    while (item != parent_[item]) {
      T parent = parent_[item];
      parent_[item] = root;
      item = parent;
    }
    return root;
  }

  // Creates the (destructive) union of the sets x and y belong to.
  void Union(T x, T y) { Link(FindSet(x), FindSet(y)); }

  // Initialization of an element: creates a singleton set containing 'item'.
  // The range [0; max) is reset if item >= max.
  T MakeSet(T item) {
    if (item >= parent_.size()) {
      // New value in parent_ should be initialized to fail_.
      const auto nitem = item > 0 ? 2 * item : 2;
      parent_.resize(nitem, fail_);
      rank_.resize(nitem);
    }
    parent_[item] = item;
    return item;
  }

  // Initialization of all elements starting from 0 to max - 1 to distinct sets.
  void MakeAllSet(T max) {
    parent_.resize(max);
    for (T item = 0; item < max; ++item) parent_[item] = item;
  }

  // For testing only.
  const T& Parent(const T& x) const { return parent_[x]; }

 private:
  // Links trees rooted in 'x' and 'y'.
  void Link(T x, T y) {
    if (x == y) return;
    if (rank_[x] > rank_[y]) {
      parent_[y] = x;
    } else {
      parent_[x] = y;
      if (rank_[x] == rank_[y]) {
        ++rank_[y];
      }
    }
  }

  UnionFind(const UnionFind&) = delete;

  UnionFind& operator=(const UnionFind&) = delete;

  std::vector<T> parent_;  // Parent nodes.
  std::vector<int> rank_;  // Rank of an element = min. depth in tree.
  T fail_;                 // Value indicating lookup failure.
};


// ArcArena is used for fast allocation of contiguous arrays of arcs.
//
// To create an arc array:
//   for each state:
//     for each arc:
//       arena.PushArc();
//     // Commits these arcs and returns pointer to them.
//     Arc *arcs = arena.GetArcs();
//
//     OR
//
//     arena.DropArcs();  // Throws away current arcs, reuse the space.
//
// The arcs returned are guaranteed to be contiguous and the pointer returned
// will never be invalidated until the arena is cleared for reuse.
//
// The contents of the arena can be released with a call to arena.Clear() after
// which the arena will restart with an initial allocation capable of holding at
// least all of the arcs requested in the last usage before Clear() making
// subsequent uses of the Arena more efficient.
//
// The max_retained_size option can limit the amount of arc space requested on
// Clear() to avoid excess growth from intermittent high usage.
template <typename Arc>
class ArcArena {
 public:
  explicit ArcArena(size_t block_size = 256, size_t max_retained_size = 1e6)
      : block_size_(block_size), max_retained_size_(max_retained_size) {
    blocks_.emplace_back(MakeSharedBlock(block_size_));
    first_block_size_ = block_size_;
    total_size_ = block_size_;
    arcs_ = blocks_.back().get();
    end_ = arcs_ + block_size_;
    next_ = arcs_;
  }

  ArcArena(const ArcArena& copy)
      : arcs_(copy.arcs_),
        next_(copy.next_),
        end_(copy.end_),
        block_size_(copy.block_size_),
        first_block_size_(copy.first_block_size_),
        total_size_(copy.total_size_),
        max_retained_size_(copy.max_retained_size_),
        blocks_(copy.blocks_) {
    NewBlock(block_size_);
  }

  void ReserveArcs(size_t n) {
    if (next_ + n < end_) return;
    NewBlock(n);
  }

  void PushArc(const Arc& arc) {
    if (next_ == end_) {
      size_t length = next_ - arcs_;
      NewBlock(length * 2);
    }
    *next_ = arc;
    ++next_;
  }

  const Arc* GetArcs() {
    const auto* arcs = arcs_;
    arcs_ = next_;
    return arcs;
  }

  void DropArcs() { next_ = arcs_; }

  size_t Size() const { return total_size_; }

  void Clear() {
    blocks_.resize(1);
    if (total_size_ > first_block_size_) {
      first_block_size_ = std::min(max_retained_size_, total_size_);
      blocks_.back() = MakeSharedBlock(first_block_size_);
    }
    total_size_ = first_block_size_;
    arcs_ = blocks_.back().get();
    end_ = arcs_ + first_block_size_;
    next_ = arcs_;
  }

 private:
  // Allocates a new block with capacity of at least n or block_size,
  // copying incomplete arc sequence from old block to new block.
  void NewBlock(size_t n) {
    const auto length = next_ - arcs_;
    const auto new_block_size = std::max(n, block_size_);
    total_size_ += new_block_size;
    blocks_.emplace_back(MakeSharedBlock(new_block_size));
    std::copy(arcs_, next_, blocks_.back().get());
    arcs_ = blocks_.back().get();
    next_ = arcs_ + length;
    end_ = arcs_ + new_block_size;
  }

  std::shared_ptr<Arc[]> MakeSharedBlock(size_t size) {
    return std::shared_ptr<Arc[]>(new Arc[size]);
  }

  Arc* arcs_;
  Arc* next_;
  const Arc* end_;
  const size_t block_size_;
  size_t first_block_size_;
  size_t total_size_;
  size_t max_retained_size_;
  std::list<std::shared_ptr<Arc[]>> blocks_;
};


// An associative container for which testing membership is faster than an STL
// set if members are restricted to an interval that excludes most non-members.
// A Key must have ==, !=, and < operators defined. Element NoKey should be a
// key that marks an uninitialized key and is otherwise unused. Find() returns
// an STL const_iterator to the match found, otherwise it equals End().
template <class Key, Key NoKey>
class CompactSet {
 public:
  using const_iterator = typename std::set<Key>::const_iterator;

  CompactSet() : min_key_(NoKey), max_key_(NoKey) {}

  CompactSet(const CompactSet&) = default;

  void Insert(Key key) {
    set_.insert(key);
    if (min_key_ == NoKey || key < min_key_) min_key_ = key;
    if (max_key_ == NoKey || max_key_ < key) max_key_ = key;
  }

  void Erase(Key key) {
    set_.erase(key);
    if (set_.empty()) {
      min_key_ = max_key_ = NoKey;
    } else if (key == min_key_) {
      ++min_key_;
    } else if (key == max_key_) {
      --max_key_;
    }
  }

  void Clear() {
    set_.clear();
    min_key_ = max_key_ = NoKey;
  }

  const_iterator Find(Key key) const {
    if (min_key_ == NoKey || key < min_key_ || max_key_ < key) {
      return set_.end();
    } else {
      return set_.find(key);
    }
  }

  bool Member(Key key) const {
    if (min_key_ == NoKey || key < min_key_ || max_key_ < key) {
      return false;  // out of range
    } else if (min_key_ != NoKey && max_key_ + 1 == min_key_ + set_.size()) {
      return true;  // dense range
    } else {
      return set_.count(key);
    }
  }

  const_iterator Begin() const { return set_.begin(); }

  const_iterator End() const { return set_.end(); }

  // All stored keys are greater than or equal to this value.
  Key LowerBound() const { return min_key_; }

  // All stored keys are less than or equal to this value.
  Key UpperBound() const { return max_key_; }

 private:
  std::set<Key> set_;
  Key min_key_;
  Key max_key_;

  void operator=(const CompactSet&) = delete;
};


}  // namespace fst

namespace {

// A stand-in for an arc: same footprint as sicada's StdArc.
struct BenchArc {
  int32_t ilabel;
  int32_t olabel;
  float weight;
  int32_t nextstate;
};

// Deterministic generator shared with the Rust side.
struct Xorshift {
  uint64_t state;
  explicit Xorshift(uint64_t seed) : state(seed) {}
  uint64_t next() {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    return state;
  }
};

struct Less {
  bool operator()(const int64_t& a, const int64_t& b) const { return a < b; }
};

}  // namespace

extern "C" {

// Heap: n inserts, then n pops, with a decrease-key on every fourth element.
uint64_t openfst_bench_heap(uint64_t n, uint64_t seed) {
  fst::Heap<int64_t, Less> heap;
  Xorshift rng(seed);
  std::vector<int> keys;
  keys.reserve(n);
  for (uint64_t i = 0; i < n; ++i) {
    keys.push_back(heap.Insert(static_cast<int64_t>(rng.next() % 1000000)));
  }
  for (uint64_t i = 0; i < n; i += 4) {
    heap.Update(keys[i], static_cast<int64_t>(rng.next() % 1000));
  }
  uint64_t checksum = 0;
  while (!heap.Empty()) checksum += static_cast<uint64_t>(heap.Pop());
  return checksum;
}

// Heap, inserts only: isolates the sift-up and growth cost.
uint64_t openfst_bench_heap_insert(uint64_t n, uint64_t seed) {
  fst::Heap<int64_t, Less> heap;
  Xorshift rng(seed);
  for (uint64_t i = 0; i < n; ++i) heap.Insert(static_cast<int64_t>(rng.next() % 1000000));
  return static_cast<uint64_t>(heap.Top());
}

// Heap, inserts then pops: isolates the sift-down cost by leaving out Update.
uint64_t openfst_bench_heap_insert_pop(uint64_t n, uint64_t seed) {
  fst::Heap<int64_t, Less> heap;
  Xorshift rng(seed);
  for (uint64_t i = 0; i < n; ++i) heap.Insert(static_cast<int64_t>(rng.next() % 1000000));
  uint64_t checksum = 0;
  while (!heap.Empty()) checksum += static_cast<uint64_t>(heap.Pop());
  return checksum;
}

// Union-find: n singletons, n-1 unions, then n lookups.
uint64_t openfst_bench_union_find(uint64_t n, uint64_t seed) {
  fst::UnionFind<int> uf(static_cast<int>(n), -1);
  uf.MakeAllSet(static_cast<int>(n));
  Xorshift rng(seed);
  for (uint64_t i = 0; i + 1 < n; ++i) {
    const int a = static_cast<int>(rng.next() % n);
    const int b = static_cast<int>(rng.next() % n);
    uf.Union(a, b);
  }
  uint64_t checksum = 0;
  for (uint64_t i = 0; i < n; ++i) checksum += static_cast<uint64_t>(uf.FindSet(static_cast<int>(i)));
  return checksum;
}

// Arc arena: `states` runs of `arcs_per_state` arcs each.
// The building half alone, for the breakdown in `diag`: the same loop without
// the read-back, so that subtracting one from the other says where the time is.
uint64_t openfst_bench_arc_arena_build(uint64_t states, uint64_t arcs_per_state) {
  fst::ArcArena<BenchArc> arena(256);
  std::vector<const BenchArc*> runs;
  runs.reserve(states);
  for (uint64_t s = 0; s < states; ++s) {
    for (uint64_t a = 0; a < arcs_per_state; ++a) {
      arena.PushArc(BenchArc{static_cast<int32_t>(a), static_cast<int32_t>(a),
                             1.0f, static_cast<int32_t>(s)});
    }
    runs.push_back(arena.GetArcs());
  }
  return reinterpret_cast<uint64_t>(runs.empty() ? nullptr : runs[0]) & 1;
}

uint64_t openfst_bench_arc_arena(uint64_t states, uint64_t arcs_per_state) {
  fst::ArcArena<BenchArc> arena(256);
  uint64_t checksum = 0;
  std::vector<const BenchArc*> runs;
  runs.reserve(states);
  for (uint64_t s = 0; s < states; ++s) {
    for (uint64_t a = 0; a < arcs_per_state; ++a) {
      arena.PushArc(BenchArc{static_cast<int32_t>(a), static_cast<int32_t>(a),
                             1.0f, static_cast<int32_t>(s)});
    }
    runs.push_back(arena.GetArcs());
  }
  for (uint64_t s = 0; s < states; ++s) {
    for (uint64_t a = 0; a < arcs_per_state; ++a) {
      checksum += static_cast<uint64_t>(runs[s][a].ilabel);
    }
  }
  return checksum;
}

// Compact set: build a set of `n` keys in a narrow interval, then probe it.
uint64_t openfst_bench_compact_set(uint64_t n, uint64_t probes, uint64_t seed) {
  fst::CompactSet<int64_t, -1> set;
  Xorshift rng(seed);
  for (uint64_t i = 0; i < n; ++i) {
    set.Insert(static_cast<int64_t>(rng.next() % (n * 2)));
  }
  uint64_t hits = 0;
  for (uint64_t i = 0; i < probes; ++i) {
    if (set.Member(static_cast<int64_t>(rng.next() % (n * 4)))) ++hits;
  }
  return hits;
}

}  // extern "C"
