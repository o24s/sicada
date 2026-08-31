//! Backing storage for a region of an FST file.
//!
//! Port of OpenFst's `mapped-file.h` / `mapped-file.cc`. A [`MappedFile`] is
//! either a memory mapping, an owned aligned allocation, or a borrowed slice.
//! Upstream's `MemoryRegion` discriminates the same three cases at run time with
//! null checks on `mmap` and `size`, which an enum expresses directly.

use memmap2::{Mmap, MmapOptions};
use std::alloc::{Layout, alloc_zeroed, dealloc};
use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};
use std::ptr::NonNull;
use std::slice;

/// Alignment required for mapping structures, in bytes.
///
/// Regions that are not aligned on a 128-bit boundary are read from the file
/// rather than mapped. Consistent with the alignment `ConstFst` and `CompactFst`
/// write.
pub const ARCH_ALIGNMENT: usize = 16;

/// A region of bytes backing part of an FST.
#[derive(Debug)]
pub enum MappedFile<'a> {
    /// A memory-mapped file region.
    Mmap(Mmap),
    /// An owned heap allocation with a guaranteed alignment.
    Owned { ptr: NonNull<u8>, layout: Layout },
    /// A borrowed region owned by someone else.
    Borrowed(&'a [u8]),
    /// An empty region. Distinguished from `Borrowed(&[])` so that a zero-sized
    /// allocation still reports itself as writable, which spares every caller a
    /// special case for empty FSTs.
    Empty,
}

// SAFETY: the `Owned` variant is the only one holding a raw pointer, and it owns
// that allocation exclusively: it is created by `allocate`, never cloned, and
// freed exactly once in `Drop`. Access goes through `&self` / `&mut self`, so
// Rust's borrow rules already prevent concurrent aliasing. `Mmap` and `&[u8]`
// are themselves `Send + Sync`.
unsafe impl Send for MappedFile<'_> {}
// SAFETY: see the `Send` impl above.
unsafe impl Sync for MappedFile<'_> {}

impl<'a> MappedFile<'a> {
    /// Allocates `size` zeroed bytes aligned to `align`.
    ///
    /// SICADA-DIVERGE: upstream hands back uninitialized memory from
    /// `operator new`. Building a `&[u8]` over uninitialized bytes is undefined
    /// behaviour in Rust regardless of what is done with it, so the region is
    /// zeroed. For the allocation sizes that matter here the allocator serves
    /// fresh zero pages from the OS, so this is not a memset in the hot path.
    pub fn allocate(size: usize, align: usize) -> io::Result<Self> {
        if size == 0 {
            return Ok(Self::Empty);
        }
        let layout = Layout::from_size_align(size, align)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;

        // SAFETY: `layout` has a non-zero size, as `alloc_zeroed` requires.
        // The returned pointer is checked for null below and is owned
        // by the `Owned` variant from here on.
        let ptr = unsafe { alloc_zeroed(layout) };
        let ptr = NonNull::new(ptr)
            .ok_or_else(|| io::Error::new(io::ErrorKind::OutOfMemory, "allocation failed"))?;

        Ok(Self::Owned { ptr, layout })
    }

    /// Allocates room for `count` values of type `T`, aligned for `T`.
    ///
    /// A `count` whose size does not fit a `usize` is an error rather than a
    /// wrap: the counts reaching here come out of files.
    pub fn allocate_type<T>(count: usize) -> io::Result<Self> {
        let size = size_of::<T>().checked_mul(count).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "allocation size overflows")
        })?;
        Self::allocate(size, align_of::<T>())
    }

    /// Wraps a region owned by someone else.
    #[inline]
    pub fn borrow(data: &'a [u8]) -> Self {
        if data.is_empty() {
            Self::Empty
        } else {
            Self::Borrowed(data)
        }
    }

    /// Maps `size` bytes of `file` starting at `pos`.
    ///
    /// Unlike [`map_or_read`](Self::map_or_read) this does not fall back to
    /// reading. `pos` need not be page-aligned; the mapping is aligned down and
    /// the result offset back up.
    pub fn map_from_file(file: &File, pos: u64, size: usize) -> io::Result<Self> {
        if size == 0 {
            return Ok(Self::Empty);
        }
        // SAFETY: mapping a file is unsound in general because another process
        // can mutate the bytes underneath the mapping. sicada treats FST files
        // as immutable while open, which is the same contract OpenFst relies on.
        let mmap = unsafe { MmapOptions::new().offset(pos).len(size).map(file)? };
        Ok(Self::Mmap(mmap))
    }

    /// Reads or maps `size` bytes at the file's current position, advancing it.
    ///
    /// `memorymap` is advisory: mapping is only attempted when the current
    /// offset meets [`ARCH_ALIGNMENT`], because a region that starts unaligned
    /// cannot be cast to the packed structures `ConstFst` and `CompactFst` store.
    /// Any failure falls back to allocating and reading.
    pub fn map_or_read(file: &mut File, memorymap: bool, size: usize) -> io::Result<Self> {
        let pos = file.stream_position()?;

        if memorymap
            && pos % ARCH_ALIGNMENT as u64 == 0
            && let Ok(mapped) = Self::map_from_file(file, pos, size)
        {
            // Mapping does not move the file cursor; upstream seeks past the
            // region for the same reason.
            file.seek(SeekFrom::Start(pos + size as u64))?;
            return Ok(mapped);
        }

        let mut owned = Self::allocate(size, ARCH_ALIGNMENT)?;
        let buf = owned
            .as_mut_slice()
            .expect("a freshly allocated region is writable");
        file.read_exact(buf)?;
        Ok(owned)
    }

    /// The region as a mutable slice, or `None` when it is not writable.
    ///
    /// Mappings are read-only, and borrowed regions belong to someone else.
    #[inline]
    pub fn as_mut_slice(&mut self) -> Option<&mut [u8]> {
        match self {
            // SAFETY: `ptr` points at `layout.size()` initialized (zeroed) bytes
            // owned solely by this value, and `&mut self` rules out any other
            // live reference into it.
            Self::Owned { ptr, layout } => {
                Some(unsafe { slice::from_raw_parts_mut(ptr.as_ptr(), layout.size()) })
            }
            Self::Empty => Some(&mut []),
            _ => None,
        }
    }
}

impl AsRef<[u8]> for MappedFile<'_> {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        match self {
            Self::Mmap(mmap) => mmap.as_ref(),
            // SAFETY: as in `as_mut_slice`, but shared; the bytes are initialized
            // and the allocation outlives the borrow of `self`.
            Self::Owned { ptr, layout } => unsafe {
                slice::from_raw_parts(ptr.as_ptr(), layout.size())
            },
            Self::Borrowed(slice) => slice,
            Self::Empty => &[],
        }
    }
}

impl std::ops::Deref for MappedFile<'_> {
    type Target = [u8];

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.as_ref()
    }
}

impl Drop for MappedFile<'_> {
    fn drop(&mut self) {
        if let Self::Owned { ptr, layout } = self {
            // SAFETY: `ptr` came from `alloc_zeroed` with exactly this layout and
            // has not been freed; `Drop` runs once.
            unsafe { dealloc(ptr.as_ptr(), *layout) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn temp_file_of(contents: &[u8]) -> NamedTempFile {
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(contents).unwrap();
        file.flush().unwrap();
        file
    }

    #[test]
    fn borrowed_regions_are_passed_through() {
        let data = b"hello openfst";
        assert_eq!(MappedFile::borrow(data).as_ref(), b"hello openfst");
    }

    #[test]
    fn allocations_are_aligned_and_writable() {
        let mut mapped = MappedFile::allocate(100, ARCH_ALIGNMENT).unwrap();
        assert_eq!(mapped.len(), 100);
        assert_eq!(mapped.as_ref().as_ptr() as usize % ARCH_ALIGNMENT, 0);

        mapped.as_mut_slice().unwrap()[0] = 42;
        assert_eq!(mapped[0], 42);
    }

    /// Regression test: the region used to come back from `alloc`, so every read
    /// of it, `len()` included, built a `&[u8]` over uninitialized bytes.
    #[test]
    fn allocations_start_zeroed() {
        for size in [1, 16, 100, 4096] {
            let mapped = MappedFile::allocate(size, ARCH_ALIGNMENT).unwrap();
            assert!(
                mapped.as_ref().iter().all(|&byte| byte == 0),
                "size {size} came back non-zero"
            );
        }
    }

    #[test]
    fn allocate_type_uses_the_types_alignment() {
        let mapped = MappedFile::allocate_type::<u64>(8).unwrap();
        assert_eq!(mapped.len(), 64);
        assert_eq!(mapped.as_ref().as_ptr() as usize % align_of::<u64>(), 0);
    }

    /// A zero-sized region must still be writable, so callers do not have to
    /// special-case an FST with no states.
    #[test]
    fn empty_regions_are_writable_and_zero_length() {
        let mut mapped = MappedFile::allocate(0, ARCH_ALIGNMENT).unwrap();
        assert!(mapped.is_empty());
        assert_eq!(mapped.as_mut_slice(), Some(&mut [][..]));

        let mut typed = MappedFile::allocate_type::<u32>(0).unwrap();
        assert_eq!(typed.as_mut_slice(), Some(&mut [][..]));
    }

    #[test]
    fn rejects_a_non_power_of_two_alignment() {
        assert!(MappedFile::allocate(16, 3).is_err());
    }

    #[test]
    fn maps_a_file_region() {
        let file = temp_file_of(b"zero copy mapping test");
        let mut handle = file.reopen().unwrap();
        let mapped = MappedFile::map_or_read(&mut handle, true, 22).unwrap();
        assert_eq!(mapped.as_ref(), b"zero copy mapping test");
        assert_eq!(handle.stream_position().unwrap(), 22);
    }

    #[test]
    fn maps_from_an_offset_that_is_not_page_aligned() {
        let file = temp_file_of(&(0u8..=255).collect::<Vec<_>>());
        let handle = file.reopen().unwrap();
        let mapped = MappedFile::map_from_file(&handle, 16, 32).unwrap();
        assert_eq!(mapped.as_ref(), &(16u8..48).collect::<Vec<_>>()[..]);
    }

    /// An unaligned start offset must fall back to reading, because the region
    /// would otherwise be uncastable to the packed on-disk structures.
    #[test]
    fn falls_back_to_reading_at_an_unaligned_offset() {
        let file = temp_file_of(b"0123456789abcdefghijklmnop");
        let mut handle = file.reopen().unwrap();
        handle.seek(SeekFrom::Start(3)).unwrap();

        let mapped = MappedFile::map_or_read(&mut handle, true, 10).unwrap();
        assert!(matches!(mapped, MappedFile::Owned { .. }));
        assert_eq!(mapped.as_ref(), b"3456789abc");
        assert_eq!(handle.stream_position().unwrap(), 13);
    }

    #[test]
    fn reading_is_used_when_mapping_is_not_requested() {
        let file = temp_file_of(b"read me please");
        let mut handle = file.reopen().unwrap();
        let mapped = MappedFile::map_or_read(&mut handle, false, 7).unwrap();
        assert!(matches!(mapped, MappedFile::Owned { .. }));
        assert_eq!(mapped.as_ref(), b"read me");
        assert_eq!(handle.stream_position().unwrap(), 7);
    }

    #[test]
    fn reading_past_the_end_of_the_file_fails() {
        let file = temp_file_of(b"short");
        let mut handle = file.reopen().unwrap();
        assert!(MappedFile::map_or_read(&mut handle, false, 100).is_err());
    }
}
