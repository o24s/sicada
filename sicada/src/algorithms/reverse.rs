use crate::algorithms::cc_visitors::SccVisitor;
use crate::algorithms::dfs_visit::dfs_visit_any;
use crate::arc::{Arc, ArcLabel, ArcStateId};
use crate::fst::{Fst, MutableFst};
use crate::properties::{
    K_COPY_PROPERTIES, K_FST_PROPERTIES, K_INITIAL_ACYCLIC, reverse_properties,
};
use crate::weight::Weight;

/// Reverses an FST. The reversed result is written to an output mutable FST.
/// If `A` transduces string `x` to `y` with weight `a`, then the reverse of `A`
/// transduces the reverse of `x` to the reverse of `y` with weight `a.reverse()`.
///
/// Typically, `a = a.reverse()` and an arc is its own reverse (e.g., for
/// `TropicalWeight` or `LogWeight`). In general, e.g., when the weights only form a
/// left or right semiring, the output arc type must match the input arc type
/// except having the reversed `Weight` type.
///
/// When `require_superinitial` is `false`, a superinitial state is not created in
/// the reversed FST iff the input FST has exactly one final state (which becomes
/// the initial state of the reversed FST) with a final weight of semiring `One`,
/// or if it does not belong to any cycle. When `require_superinitial` is `true`, a
/// superinitial state is always created.
pub fn reverse<FromArc, ToArc, F1, F2>(ifst: &F1, ofst: &mut F2, require_superinitial: bool)
where
    FromArc: Arc,
    ToArc: Arc<
            Label = FromArc::Label,
            StateId = FromArc::StateId,
            Weight = <FromArc::Weight as Weight>::ReverseWeight,
        >,
    F1: Fst<FromArc>,
    F2: MutableFst<ToArc>,
{
    ofst.delete_all_states();
    ofst.set_input_symbols(ifst.input_symbols());
    ofst.set_output_symbols(ifst.output_symbols());

    if let Some(num_states) = ifst.num_states_if_known() {
        ofst.reserve_states(num_states + 1);
    }

    let istart = ifst.start();
    let mut ostart_opt = None;
    let mut offset = 0;
    let mut dfs_iprops = 0;
    let mut dfs_oprops = 0;

    if !require_superinitial {
        for s in ifst.states() {
            if !ifst.final_weight(s).is_member() || ifst.final_weight(s) == FromArc::Weight::zero()
            {
                continue;
            }
            if ostart_opt.is_some() {
                ostart_opt = None;
                break;
            } else {
                ostart_opt = Some(s);
            }
        }

        // SICADA-DIVERGE: a weight that is not a member of its semiring, such
        // as `no_weight()`, which is NaN for the float weights, is not a final
        // weight, so a state carrying one is not a final state. Upstream tests
        // only against `Zero`, and NaN is not equal to it, so it would take
        // such a state as the single final one and put the invalid weight into
        // the output.
        if let Some(ostart) = ostart_opt
            && ifst.final_weight(ostart) != FromArc::Weight::one()
        {
            let mut scc = Vec::new();
            {
                let mut scc_visitor =
                    SccVisitor::new(ifst, Some(&mut scc), None, None, &mut dfs_iprops);
                dfs_visit_any(ifst, &mut scc_visitor);
            }

            let ostart_idx = ostart.as_usize();
            if scc.len() > ostart_idx {
                let comp = scc[ostart_idx];
                let count = scc.iter().filter(|&&c| c == comp).count();
                if count > 1 {
                    ostart_opt = None;
                }
            }

            if ostart_opt.is_some() {
                for arc in ifst.arcs(ostart) {
                    if arc.nextstate() == ostart {
                        ostart_opt = None;
                        break;
                    }
                }
            }

            if ostart_opt.is_some() {
                dfs_oprops |= K_INITIAL_ACYCLIC;
            }
        }
    }

    let ostart = match ostart_opt {
        Some(s) => s,
        None => {
            offset = 1;
            ofst.add_state() // Super-initial requested or needed (State 0)
        }
    };

    for is in ifst.states() {
        let os_idx = is.as_usize() + offset;
        let os = ToArc::StateId::from_usize(os_idx);

        while ofst.num_states() <= os_idx {
            ofst.add_state();
        }

        if Some(is) == istart {
            ofst.set_final(os, ToArc::Weight::one());
        }

        let weight = ifst.final_weight(is);
        if weight.is_member() && weight != FromArc::Weight::zero() && offset == 1 {
            let oarc = ToArc::new(
                ToArc::Label::epsilon(),
                ToArc::Label::epsilon(),
                weight.reverse(),
                os,
            );
            // 0 is always the super-initial state when offset == 1
            ofst.add_arc(ToArc::StateId::from_usize(0), oarc);
        }

        for iarc in ifst.arcs(is) {
            let nos_idx = iarc.nextstate().as_usize() + offset;
            let nos = ToArc::StateId::from_usize(nos_idx);

            let mut rev_weight = iarc.weight().reverse();
            if offset == 0 && nos == ostart {
                rev_weight = ifst.final_weight(ostart).reverse().times(&rev_weight);
            }

            let oarc = ToArc::new(iarc.ilabel(), iarc.olabel(), rev_weight, os);

            while ofst.num_states() <= nos_idx {
                ofst.add_state();
            }
            ofst.add_arc(nos, oarc);
        }
    }

    ofst.set_start(ostart);

    if offset == 0 && Some(ostart) == istart {
        ofst.set_final(ostart, ifst.final_weight(ostart).reverse());
    }

    let iprops = ifst.properties(K_COPY_PROPERTIES, false) | dfs_iprops;
    let oprops = ofst.properties(K_FST_PROPERTIES, false) | dfs_oprops;

    ofst.set_properties(
        reverse_properties(iprops, offset == 1) | oprops,
        K_FST_PROPERTIES,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::test_support::{Rng, paths, random_acyclic_fst, sorted};
    use crate::arc::StdArc;
    use crate::float_weight::TropicalWeight;
    use crate::fst::{ExpandedFst as _, Fst, MutableFst};
    use crate::fsts::vector_fst::StdVectorFst;
    use crate::weight::Weight;

    #[test]
    fn test_reverse_without_superinitial() {
        let mut ifst = StdVectorFst::new();
        let s0 = ifst.add_state();
        let s1 = ifst.add_state();
        let s2 = ifst.add_state();

        ifst.set_start(s0);
        ifst.set_final(s2, TropicalWeight::one());

        // 0 -> 1 -> 2
        ifst.add_arc(s0, StdArc::new(1, 1, TropicalWeight::one(), s1));
        ifst.add_arc(s1, StdArc::new(2, 2, TropicalWeight::one(), s2));

        let mut ofst = StdVectorFst::new();
        // require_superinitial = false
        // Final state 2 has weight `One` and no cycles.
        // It shouldn't need a super-initial state.
        reverse(&ifst, &mut ofst, false);

        assert_eq!(ofst.num_states(), 3);

        let start = ofst.start().unwrap();
        // The original final state `s2` had index 2. Without superinitial, offset=0.
        assert_eq!(start.as_usize(), 2);
        assert_eq!(ofst.final_weight(s0), TropicalWeight::one());
    }

    #[test]
    fn test_reverse_with_superinitial() {
        let mut ifst = StdVectorFst::new();
        let s0 = ifst.add_state();
        let s1 = ifst.add_state();

        ifst.set_start(s0);
        ifst.set_final(s1, TropicalWeight::one());

        ifst.add_arc(s0, StdArc::new(1, 1, TropicalWeight::one(), s1));

        let mut ofst = StdVectorFst::new();
        // require_superinitial = true
        reverse(&ifst, &mut ofst, true);

        // A super-initial state (0) is prepended.
        assert_eq!(ofst.num_states(), 3);
        let start = ofst.start().unwrap();
        assert_eq!(start.as_usize(), 0);

        // The old final state `s1` is mapped to state 1 + offset = 2
        // So a super-initial arc goes from 0 -> 2
        let mut arc_iter = ofst.arcs(start);
        let arc = arc_iter.next().unwrap();
        assert_eq!(arc.nextstate().as_usize(), 2);
    }

    /// What reversing means: the reversed FST transduces the reverse of every
    /// string the input does, with the same weight, at least for the tropical
    /// semiring, where reversing a weight leaves it alone.
    fn assert_reverses(ifst: &StdVectorFst, require_superinitial: bool) {
        let mut ofst = StdVectorFst::new();
        reverse(ifst, &mut ofst, require_superinitial);

        // The reversed FST has an epsilon arc out of any superinitial state, so
        // it needs one more step to reach the same paths.
        let want = sorted(
            paths(ifst, 6)
                .into_iter()
                .map(|(mut i, mut o, w)| {
                    i.reverse();
                    o.reverse();
                    (i, o, w)
                })
                .collect(),
        );
        // Epsilon arcs from the superinitial state contribute no labels, so
        // they drop out of the comparison once removed.
        let got = sorted(
            paths(&ofst, 7)
                .into_iter()
                .map(|(i, o, w)| {
                    (
                        i.into_iter().filter(|&l| l != 0).collect(),
                        o.into_iter().filter(|&l| l != 0).collect(),
                        w,
                    )
                })
                .collect(),
        );
        assert_eq!(got, want, "superinitial={require_superinitial}");
    }

    #[test]
    fn reversing_transduces_the_reversed_strings() {
        let mut rng = Rng::new(0x9E37_79B9);
        for _ in 0..200 {
            let fst = random_acyclic_fst(&mut rng, 5);
            assert_reverses(&fst, true);
            assert_reverses(&fst, false);
        }
    }
}
