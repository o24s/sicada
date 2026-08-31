// TODO(porting-iteration): drop this once the porting iteration reaches this file.
#![allow(unused)]

#[cfg(not(debug_assertions))]
use std::cell::UnsafeCell;
#[cfg(debug_assertions)]
use std::cell::{Ref, RefCell, RefMut};
use std::ops::{Deref, DerefMut};

#[cfg(debug_assertions)]
pub struct FastCell<T>(RefCell<T>);

#[cfg(not(debug_assertions))]
pub struct FastCell<T>(UnsafeCell<T>);

impl<T> FastCell<T> {
    #[inline(always)]
    pub fn new(value: T) -> Self {
        #[cfg(debug_assertions)]
        return Self(RefCell::new(value));

        #[cfg(not(debug_assertions))]
        return Self(UnsafeCell::new(value));
    }

    #[inline(always)]
    pub fn borrow(&self) -> FastRef<'_, T> {
        #[cfg(debug_assertions)]
        return FastRef(self.0.borrow());

        #[cfg(not(debug_assertions))]
        return FastRef(unsafe { &*self.0.get() });
    }

    #[inline(always)]
    pub fn borrow_mut(&self) -> FastRefMut<'_, T> {
        #[cfg(debug_assertions)]
        return FastRefMut(self.0.borrow_mut());

        #[cfg(not(debug_assertions))]
        return FastRefMut(unsafe { &mut *self.0.get() });
    }
}

#[cfg(debug_assertions)]
pub struct FastRef<'a, T>(Ref<'a, T>);

#[cfg(not(debug_assertions))]
pub struct FastRef<'a, T>(&'a T);

impl<'a, T> Deref for FastRef<'a, T> {
    type Target = T;
    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[cfg(debug_assertions)]
pub struct FastRefMut<'a, T>(RefMut<'a, T>);

#[cfg(not(debug_assertions))]
pub struct FastRefMut<'a, T>(&'a mut T);

impl<'a, T> Deref for FastRefMut<'a, T> {
    type Target = T;
    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'a, T> DerefMut for FastRefMut<'a, T> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
