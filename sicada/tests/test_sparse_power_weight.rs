use sicada::weight::{Divide, DivideType, Weight};
use sicada::weights::float_weight::TropicalWeight;
use sicada::weights::sparse_power_weight::SparsePowerWeight;
use sicada::weights::sparse_tuple_weight::SparseTupleWeight;

type SparsePowerTropical = SparsePowerWeight<TropicalWeight, i64>;

#[test]
fn test_sparse_power_weight_parse_display() {
    let text = "0,1,5,3,10";
    let w: SparsePowerTropical = text.parse().expect("Failed to parse SparsePowerWeight");

    assert_eq!(w.to_string(), text);
    assert_eq!(w.inner.size(), 2);

    // Default 0.0, with key 1 at 5.0, key 2 left at the default and key 3 at 10.0.
    assert_eq!(w.inner.value(1).value(), 5.0);
    assert_eq!(w.inner.value(2).value(), 0.0);
    assert_eq!(w.inner.value(3).value(), 10.0);
}

#[test]
fn test_sparse_power_weight_plus() {
    // ⊕ over the tropical semiring is `min`.
    let mut inner1 = SparseTupleWeight::new(TropicalWeight::zero()); // Zero = Infinity
    inner1.set_value(1, TropicalWeight(2.0));
    inner1.set_value(2, TropicalWeight(4.0));
    let w1 = SparsePowerTropical::new(inner1);

    let mut inner2 = SparseTupleWeight::new(TropicalWeight::zero());
    inner2.set_value(2, TropicalWeight(5.0));
    inner2.set_value(3, TropicalWeight(6.0));
    let w2 = SparsePowerTropical::new(inner2);

    let w3 = w1.plus(&w2);

    // default: min(Inf, Inf) = Inf
    // 1: min(2.0, Inf) = 2.0
    // 2: min(4.0, 5.0) = 4.0
    // 3: min(Inf, 6.0) = 6.0
    assert_eq!(w3.inner.default_value(), &TropicalWeight::zero());
    assert_eq!(w3.inner.value(1).value(), 2.0);
    assert_eq!(w3.inner.value(2).value(), 4.0);
    assert_eq!(w3.inner.value(3).value(), 6.0);
}

#[test]
fn test_sparse_power_weight_times() {
    // ⊗ adds.
    let mut inner1 = SparseTupleWeight::new(TropicalWeight::one()); // One = 0.0
    inner1.set_value(1, TropicalWeight(2.0));
    inner1.set_value(2, TropicalWeight(4.0));
    let w1 = SparsePowerTropical::new(inner1);

    let mut inner2 = SparseTupleWeight::new(TropicalWeight::one());
    inner2.set_value(2, TropicalWeight(5.0));
    inner2.set_value(3, TropicalWeight(6.0));
    let w2 = SparsePowerTropical::new(inner2);

    let w3 = w1.times(&w2);

    // default: 0.0 + 0.0 = 0.0
    // 1: 2.0 + 0.0 = 2.0
    // 2: 4.0 + 5.0 = 9.0
    // 3: 0.0 + 6.0 = 6.0
    assert_eq!(w3.inner.default_value(), &TropicalWeight::one());
    assert_eq!(w3.inner.value(1).value(), 2.0);
    assert_eq!(w3.inner.value(2).value(), 9.0);
    assert_eq!(w3.inner.value(3).value(), 6.0);
}

#[test]
fn test_sparse_power_weight_dot_product() {
    let mut inner1 = SparseTupleWeight::new(TropicalWeight::one()); // default: 0.0
    inner1.set_value(1, TropicalWeight(2.0));
    inner1.set_value(2, TropicalWeight(4.0));
    let w1 = SparsePowerTropical::new(inner1);

    let mut inner2 = SparseTupleWeight::new(TropicalWeight::one());
    inner2.set_value(2, TropicalWeight(5.0));
    inner2.set_value(3, TropicalWeight(6.0));
    let w2 = SparsePowerTropical::new(inner2);

    // The dot product takes ⊗ of the two and then folds ⊕ over the elements
    // that are not the default.
    // ⊗ gives (1: 2.0, 2: 9.0, 3: 6.0), with 0.0 as the default.
    // Folding ⊕, which is `min`, from zero (infinity):
    // -> min(Inf, 2.0) = 2.0
    // -> min(2.0, 9.0) = 2.0
    // -> min(2.0, 6.0) = 2.0
    let dot = w1.dot_product(&w2);
    assert_eq!(dot.value(), 2.0);
}

#[test]
fn test_sparse_power_weight_divide() {
    // Division subtracts.
    let text1 = "0,1,5,2,9";
    let w1: SparsePowerTropical = text1.parse().unwrap();

    let text2 = "0,1,2,2,5";
    let w2: SparsePowerTropical = text2.parse().unwrap();

    let w3 = w1.divide(&w2, DivideType::Any);

    // 1: 5.0 - 2.0 = 3.0
    // 2: 9.0 - 5.0 = 4.0
    assert_eq!(w3.inner.value(1).value(), 3.0);
    assert_eq!(w3.inner.value(2).value(), 4.0);
    assert_eq!(w3.to_string(), "0,1,3,2,4");
}

#[test]
fn test_sparse_power_weight_approx_equal() {
    // A difference of 1e-3, which f32 holds without rounding it away.
    let text1 = "0,1,5.001,2,9";
    let w1: SparsePowerTropical = text1.parse().unwrap();

    let text2 = "0,1,5,2,9";
    let w2: SparsePowerTropical = text2.parse().unwrap();

    // At a tolerance of 1e-2 that difference is below it, so the two are equal.
    assert!(w1.approx_equal(&w2, 1e-2));

    // At 1e-4 it is above, so they are not.
    assert!(!w1.approx_equal(&w2, 1e-4));
}
