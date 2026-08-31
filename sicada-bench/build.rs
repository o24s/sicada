use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=cpp/openfst_shim.cc");
    println!("cargo:rerun-if-changed=cpp/openfst_algo_shim.cc");
    println!("cargo:rerun-if-env-changed=OPENFST_BUILD_DIR");

    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .opt_level(3)
        // The shim is upstream's code verbatim; its warnings are not ours to fix.
        .flag_if_supported("-Wno-sign-compare")
        .warnings(false)
        .file("cpp/openfst_shim.cc")
        .compile("openfst_shim");

    // The algorithm benchmarks link against a real build of OpenFst, which has
    // to be made once by hand: it needs cmake and downloads abseil. Without it
    // the data-structure benchmarks still build and run.
    let Ok(build_dir) = std::env::var("OPENFST_BUILD_DIR") else {
        return;
    };
    let build_dir = PathBuf::from(build_dir);
    // The vendored upstream sources. A clone that never ran `git submodule
    // update` does not have them, which is not an error here: it skips the
    // algorithm benchmarks, the same as `OPENFST_BUILD_DIR` being unset.
    let Ok(source_dir) = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../vendor/openfst")
        .canonicalize()
    else {
        println!(
            "cargo:warning=OPENFST_BUILD_DIR is set but the vendor/openfst submodule is \
             not checked out; skipping the algorithm benchmarks"
        );
        return;
    };
    let absl_dir = build_dir.join("_deps/abseil-cpp-src");

    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .opt_level(3)
        .include(&source_dir)
        .include(&absl_dir)
        .flag_if_supported("-Wno-sign-compare")
        .warnings(false)
        .file("cpp/openfst_algo_shim.cc")
        .compile("openfst_algo_shim");

    println!("cargo:rustc-cfg=openfst_algorithms");
    println!(
        "cargo:rustc-link-search=native={}",
        build_dir.join("openfst/lib").display()
    );
    println!("cargo:rustc-link-lib=static=fst");
    // Every abseil archive the build produced; which ones OpenFst actually
    // pulls in varies with its version, so the list is discovered rather than
    // written down.
    let absl_build = build_dir.join("_deps/abseil-cpp-build/absl");
    let mut libs = Vec::new();
    collect_archives(&absl_build, &mut libs);
    for dir in libs
        .iter()
        .map(|(dir, _)| dir)
        .collect::<std::collections::BTreeSet<_>>()
    {
        println!("cargo:rustc-link-search=native={}", dir.display());
    }
    // Repeated so that the linker can resolve cycles between the archives
    // without a topological order.
    for _ in 0..3 {
        for (_, name) in &libs {
            println!("cargo:rustc-link-lib=static={name}");
        }
    }
    println!("cargo:rustc-link-lib=dylib=stdc++");
}

/// Every `libfoo.a` under `dir`, as (directory, `foo`).
fn collect_archives(dir: &std::path::Path, out: &mut Vec<(PathBuf, String)>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_archives(&path, out);
        } else if let Some(name) = path.file_name().and_then(|n| n.to_str())
            && let Some(stem) = name.strip_prefix("lib").and_then(|n| n.strip_suffix(".a"))
        {
            out.push((path.parent().unwrap().to_path_buf(), stem.to_string()));
        }
    }
}
