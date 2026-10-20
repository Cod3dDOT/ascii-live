//! Dependency-free engine benchmarks (`harness = false`).
//!
//!   cargo bench                          # run everything
//!   cargo bench -- frame                 # only cases containing "frame"
//!   cargo bench -- --save NAME           # save results as baseline NAME
//!   cargo bench -- --merge NAME          # save, keeping the best per case
//!   cargo bench -- --baseline NAME       # compare this run against NAME
//!   cargo bench -- --compare OLD NEW     # compare two saved baselines
//!   cargo bench -- --levels              # SIMD levels this CPU supports
//!
//! Reports wall time (min/median) and user-space instructions via
//! `perf_event_open` (works at perf_event_paranoid=2). Prefer instruction
//! deltas for small changes. `SITU_SIMD=avx2|sse2|...` pins the SIMD level.
//! Baselines: target/bench/NAME.tsv (or $BENCH_DIR).

use std::collections::BTreeMap;
use std::hint::black_box;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use situ::Config;
use situ::render::{Layout, Renderer};
use situ::sim::Sim;
use situ::workload;

/// User-space instruction counter for this thread (None if unavailable).
struct Counter(i32);

impl Counter {
    fn new() -> Option<Self> {
        // perf_event_attr, PERF_ATTR_SIZE_VER0 layout (64 bytes).
        #[repr(C)]
        struct Attr {
            kind: u32,
            size: u32,
            config: u64,
            sample_period: u64,
            sample_type: u64,
            read_format: u64,
            flags: u64,
            wakeup_events: u32,
            bp_type: u32,
            config1: u64,
        }
        let attr = Attr {
            kind: 0, // PERF_TYPE_HARDWARE
            size: 64,
            config: 1, // PERF_COUNT_HW_INSTRUCTIONS
            sample_period: 0,
            sample_type: 0,
            read_format: 0,
            flags: 1 | 1 << 5 | 1 << 6, // disabled, exclude_kernel, exclude_hv
            wakeup_events: 0,
            bp_type: 0,
            config1: 0,
        };
        // SAFETY: valid attr pointer; counts this thread on any CPU.
        let fd = unsafe { libc::syscall(libc::SYS_perf_event_open, &attr, 0, -1, -1, 0) };
        (fd >= 0).then_some(Counter(fd as i32))
    }

    fn ioctl(&self, req: libc::c_ulong) {
        // SAFETY: perf ioctl on our own fd.
        unsafe { libc::ioctl(self.0, req, 0) };
    }

    fn read(&self) -> u64 {
        let mut v = 0u64;
        // SAFETY: reads one u64 into `v`.
        unsafe { libc::read(self.0, (&raw mut v).cast(), 8) };
        v
    }
}

/// Accumulates time and instructions over explicitly measured regions.
struct Meter {
    counter: Option<Counter>,
    time: Duration,
    instr: u64,
}

impl Meter {
    fn region<R>(&mut self, f: impl FnOnce() -> R) -> R {
        const ENABLE: libc::c_ulong = 0x2400;
        const DISABLE: libc::c_ulong = 0x2401;
        const RESET: libc::c_ulong = 0x2403;
        if let Some(c) = &self.counter {
            c.ioctl(RESET);
            c.ioctl(ENABLE);
        }
        let t = Instant::now();
        let r = f();
        self.time += t.elapsed();
        if let Some(c) = &self.counter {
            c.ioctl(DISABLE);
            self.instr += c.read();
        }
        r
    }
}

#[derive(Clone, Copy)]
struct Stats {
    min: f64,
    median: f64,
    /// Instructions per unit (0 if counters are unavailable).
    instr: f64,
}

/// Repeats `f` for ~`budget`; `f` measures its own regions on the meter and
/// returns how many units of work it did. Time is reported as `scale` per unit.
fn measure(budget: Duration, scale: f64, mut f: impl FnMut(&mut Meter) -> usize) -> Stats {
    let mut meter = Meter {
        counter: Counter::new(),
        time: Duration::ZERO,
        instr: 0,
    };
    f(&mut meter); // warm-up
    let (mut times, mut instrs) = (Vec::new(), Vec::new());
    let start = Instant::now();
    while start.elapsed() < budget || times.len() < 5 {
        (meter.time, meter.instr) = (Duration::ZERO, 0);
        let n = f(&mut meter).max(1) as f64;
        times.push(meter.time.as_secs_f64() * scale / n);
        instrs.push(meter.instr as f64 / n);
    }
    times.sort_by(f64::total_cmp);
    instrs.sort_by(f64::total_cmp);
    Stats {
        min: times[0],
        median: times[times.len() / 2],
        instr: instrs[instrs.len() / 2],
    }
}

fn bytes(px: &mut [u32]) -> &mut [u8] {
    // SAFETY: plain-old-data view of an aligned u32 buffer.
    unsafe { std::slice::from_raw_parts_mut(px.as_mut_ptr().cast(), px.len() * 4) }
}

/// Screen sizes x cell densities worth tracking.
const SCREENS: [(&str, usize, usize, f32); 3] = [
    ("1080p", 1920, 1080, 54.0),
    ("1600p", 2560, 1600, 54.0),
    ("4k-dense", 3840, 2160, 240.0),
];

/// One workload episode; only the frames are measured. Returns frames.
fn episode(m: &mut Meter, cfg: &Config, (w, h): (usize, usize), paint: bool) -> usize {
    let layout = Layout::new(w, h, cfg.style.cell_px(h));
    let (cols, rows) = (layout.cols, layout.rows);
    let rest = cfg.style.generate(cols, rows);
    let mut sim = Sim::new(cols, rows, rest, cfg.params.clone());
    let mut r = Renderer::new(layout, &cfg.style);
    let s = r.add_buffer();
    let mut px = vec![0u32; w * h];
    r.sync(&mut sim);
    r.paint(sim.level(), s, bytes(&mut px), w * 4);
    let frames = m.region(|| {
        workload::episode(&mut sim, cfg.fps, cols, rows, |sim, _| {
            if paint {
                r.sync(sim);
                r.paint(sim.level(), s, bytes(&mut px), w * 4);
                black_box(r.take_damage());
            }
        })
    });
    black_box(&px);
    frames
}

type Case = (String, &'static str, Box<dyn Fn(Duration) -> Stats>);

fn cases() -> Vec<Case> {
    let mut cases: Vec<Case> = Vec::new();
    for (name, w, h, rows) in SCREENS {
        let cfg = move || {
            let mut c = Config::default();
            c.style.rows = rows;
            c
        };
        cases.push((
            format!("field/generate/{name}"),
            "ms",
            Box::new(move |b| {
                let c = cfg();
                let l = Layout::new(w, h, c.style.cell_px(h));
                measure(b, 1e3, |m| {
                    m.region(|| drop(black_box(c.style.generate(l.cols, l.rows))));
                    1
                })
            }),
        ));
        cases.push((
            format!("renderer/new/{name}"),
            "ms",
            Box::new(move |b| {
                let c = cfg();
                measure(b, 1e3, |m| {
                    let l = Layout::new(w, h, c.style.cell_px(h));
                    m.region(|| drop(black_box(Renderer::new(l, &c.style))));
                    1
                })
            }),
        ));
        for (kind, paint) in [("sim", false), ("sim+paint", true)] {
            cases.push((
                format!("frame/{kind}/{name}"),
                "us/frame",
                Box::new(move |b| {
                    let c = cfg();
                    measure(b, 1e6, |m| episode(m, &c, (w, h), paint))
                }),
            ));
        }
    }
    cases
}

/// Baseline directory: `BENCH_DIR`, else target/bench.
fn dir() -> PathBuf {
    std::env::var_os("BENCH_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("../bench"))
}

fn load(name: &str) -> BTreeMap<String, Stats> {
    let parse = |l: &str| {
        let mut f = l.split('\t');
        let k = f.next()?.to_string();
        let mut n = || f.next()?.parse::<f64>().ok();
        let (min, median, instr) = (n()?, n()?, n()?);
        Some((k, Stats { min, median, instr }))
    };
    std::fs::read_to_string(dir().join(format!("{name}.tsv")))
        .unwrap_or_default()
        .lines()
        .filter_map(parse)
        .collect()
}

fn save(name: &str, results: &BTreeMap<String, Stats>, keep_best: bool) {
    let mut all = load(name);
    for (k, s) in results {
        let e = all.entry(k.clone()).or_insert(*s);
        *e = if keep_best {
            Stats {
                min: e.min.min(s.min),
                median: e.median.min(s.median),
                instr: e.instr.min(s.instr),
            }
        } else {
            *s
        };
    }
    let text: String = all
        .iter()
        .map(|(k, s)| format!("{k}\t{}\t{}\t{}\n", s.min, s.median, s.instr))
        .collect();
    std::fs::create_dir_all(dir()).unwrap();
    std::fs::write(dir().join(format!("{name}.tsv")), text).unwrap();
}

fn pct(new: f64, old: Option<f64>) -> String {
    old.filter(|&o| o > 0.0 && new > 0.0)
        .map(|o| format!("{:+.1}%", (new / o - 1.0) * 100.0))
        .unwrap_or_default()
}

fn header() {
    println!(
        "{:<26} {:>10} {:>10} {:<9} {:>10} {:>8} {:>8}",
        "case", "min", "median", "unit", "Kinstr", "Δmin", "Δinstr"
    );
}

fn row(name: &str, unit: &str, s: Stats, base: Option<&Stats>) {
    println!(
        "{name:<26} {:>10.3} {:>10.3} {unit:<9} {:>10.1} {:>8} {:>8}",
        s.min,
        s.median,
        s.instr / 1e3,
        pct(s.min, base.map(|b| b.min)),
        pct(s.instr, base.map(|b| b.instr)),
    );
}

fn main() {
    let mut args = std::env::args().skip(1).filter(|a| a != "--bench");
    let (mut out, mut keep_best, mut baseline, mut filter) = (None, false, None, None);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--save" => out = args.next(),
            "--merge" => (out, keep_best) = (args.next(), true),
            "--baseline" => baseline = args.next(),
            "--levels" => {
                let names: Vec<_> = situ::sim::simd_levels().iter().map(|l| l.0).collect();
                return println!("{}", names.join(" "));
            }
            "--compare" => {
                let (old, new) = (load(&args.next().unwrap()), load(&args.next().unwrap()));
                let units: BTreeMap<_, _> = cases().into_iter().map(|(n, u, _)| (n, u)).collect();
                header();
                for (k, s) in &new {
                    row(k, units.get(k).copied().unwrap_or(""), *s, old.get(k));
                }
                return;
            }
            _ => filter = Some(a),
        }
    }
    let budget = Duration::from_millis(
        std::env::var("BENCH_MS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1500),
    );
    if Counter::new().is_none() {
        eprintln!("note: perf counters unavailable (perf_event_paranoid > 2?); Kinstr = 0");
    }
    let base = baseline.as_deref().map(load).unwrap_or_default();
    let mut results = BTreeMap::new();
    header();
    for (name, unit, run) in cases() {
        if filter.as_ref().is_some_and(|f| !name.contains(f.as_str())) {
            continue;
        }
        let s = run(budget);
        row(&name, unit, s, base.get(&name));
        results.insert(name, s);
    }
    if let Some(name) = out {
        save(&name, &results, keep_best);
        eprintln!("saved baseline '{name}'");
    }
}
