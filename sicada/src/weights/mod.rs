pub mod error_weight;
pub mod expectation_weight;
pub mod float_weight;
pub mod lexicographic_weight;
pub mod pair_weight;
pub mod power_weight;
pub mod power_weight_mappers;
pub mod product_weight;
pub mod set_weight;
pub mod signed_log_weight;
pub mod sparse_power_weight;
pub mod sparse_tuple_weight;
pub mod string_weight;
pub mod tuple_weight;
pub mod union_weight;

/// -log(e^-x + e^-y) = x - LogPosExp(y - x), assuming y >= x.
#[inline(always)]
pub(crate) fn log_pos_exp(x: f64) -> f64 {
    // NB: NaN values are allowed.
    (-x).exp().ln_1p()
}

/// -log(e^-x - e^-y) = x - LogNegExp(y - x), assuming y >= x.
#[inline(always)]
pub(crate) fn log_neg_exp(x: f64) -> f64 {
    // NB: NaN values are allowed.
    (-(-x).exp()).ln_1p()
}

/// a +_log b = -log(e^-a + e^-b) = KahanLogSum(a, b, ...).
/// Kahan compensated summation provides an error bound that is
/// independent of the number of addends. Assumes b >= a;
/// c is the compensation.
#[inline(always)]
pub(crate) fn kahan_log_sum(a: f64, b: f64, c: &mut f64) -> f64 {
    let y = -log_pos_exp(b - a) - *c;
    let t = a + y;
    *c = (t - a) - y;
    t
}

/// a -_log b = -log(e^-a - e^-b) = KahanLogDiff(a, b, ...).
/// Kahan compensated summation provides an error bound that is
/// independent of the number of addends. Assumes b > a;
/// c is the compensation.
#[inline(always)]
pub(crate) fn kahan_log_diff(a: f64, b: f64, c: &mut f64) -> f64 {
    let y = -log_neg_exp(b - a) - *c;
    let t = a + y;
    *c = (t - a) - y;
    t
}
