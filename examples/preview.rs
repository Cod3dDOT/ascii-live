//! Offline renderer for development and profiling; no compositor needed.
//!
//!   cargo run --release --example preview -- OUT.ppm [--size WxH]
//!       [--stir [--repeat N] [--frames DIR]] [--atlas] [situ options]
//!
//! Default: render the rest image. `--stir`: replay `situ::workload`
//! until settled, printing per-frame cost and damage. `--frames DIR`: write
//! every frame (with cursor). `--atlas`: dump all glyph sprites.

use std::io::Write;
use std::time::Instant;

use situ::cli::{self, Cmd};
use situ::render::{Layout, Renderer};
use situ::sim::Sim;
use situ::workload;

fn die(msg: &str) -> ! {
    eprintln!("preview: {msg}");
    std::process::exit(1)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .unwrap_or_else(|| die("usage: preview OUT.ppm [options]"));
    let (mut size, mut stir, mut repeat, mut atlas, mut frames_dir) =
        ((1920, 1080), false, 1, false, None);
    let mut rest = Vec::new();
    while let Some(a) = args.next() {
        match a.as_str() {
            "--size" => {
                let v = args.next().unwrap_or_default();
                let parse = |s: &str| s.parse().unwrap_or_else(|_| die("--size WxH"));
                let (w, h) = v.split_once('x').unwrap_or_else(|| die("--size WxH"));
                size = (parse(w), parse(h));
            }
            "--stir" => stir = true,
            "--repeat" => repeat = args.next().and_then(|v| v.parse().ok()).unwrap_or(1),
            "--frames" => frames_dir = args.next(),
            "--atlas" => atlas = true,
            _ => rest.push(a),
        }
    }
    let cfg = match cli::parse(rest) {
        Ok(Cmd::Run(c)) => c,
        Ok(Cmd::Help) => return print!("{}", cli::HELP),
        Err(e) => die(&e),
    };
    let (w, h) = size;

    if atlas {
        let r = Renderer::new(Layout::new(1, 1, 48), &cfg.style);
        let (aw, ah, img) = r.atlas_image();
        return write_ppm(&path, &img, aw, ah);
    }

    let layout = Layout::new(w, h, cfg.style.cell_px(h));
    let (cols, rows, cell, ox, oy) = (layout.cols, layout.rows, layout.cell, layout.ox, layout.oy);
    let mut sim = Sim::new(
        cols,
        rows,
        cfg.style.generate(cols, rows),
        cfg.params.clone(),
    );
    let mut r = Renderer::new(layout, &cfg.style);
    let mut px = vec![0u32; w * h];
    let state = r.add_buffer();
    r.sync(&mut sim);
    r.paint(sim.level(), state, bytes_mut(&mut px), w * 4);
    r.take_damage();
    let cursor = move |t: f32| {
        let (cx, cy) = workload::pointer(t, cols, rows);
        let at = |v: f32, o: isize| (o as f32 + v * cell as f32) as isize;
        (at(cx, ox), at(cy, oy))
    };
    let save_frame = |dir: &str, f: usize, px: &[u32], t: f32| {
        let mut img = px.to_vec();
        let (x, y) = cursor(t);
        draw_cursor(&mut img, w, h, x, y, cell);
        write_ppm(&format!("{dir}/{f:04}.ppm"), &img, w, h);
    };
    if let Some(dir) = &frames_dir {
        std::fs::create_dir_all(dir).unwrap_or_else(|e| die(&format!("{dir}: {e}")));
        // Frame 0 is the rest image: the settled last frame matches it.
        save_frame(dir, 0, &px, 0.0);
    }

    for run in 0..if stir { repeat } else { 0 } {
        let (mut sim_t, mut paint_t, mut damaged, mut rects) = (0.0, 0.0, 0usize, 0usize);
        let mut t_last = Instant::now();
        let mut f = 1;
        let frames = workload::episode(&mut sim, cfg.fps, cols, rows, |sim, t| {
            let t1 = Instant::now();
            sim_t += (t1 - t_last).as_secs_f64();
            r.sync(sim);
            r.paint(sim.level(), state, bytes_mut(&mut px), w * 4);
            let d = r.take_damage();
            paint_t += t1.elapsed().as_secs_f64();
            damaged += d.iter().map(|d| (d.2 * d.3) as usize).sum::<usize>();
            rects += d.len();
            if run == 0
                && let Some(dir) = &frames_dir
            {
                save_frame(dir, f, &px, t);
            }
            f += 1;
            t_last = Instant::now();
        });
        let n = frames as f64;
        eprintln!(
            "{cols}x{rows} cells: {frames} frames ({:.1}s); per frame: sim {:.3}ms, paint {:.3}ms, \
             damage {:.1}% of pixels in {:.0} rects",
            n / f64::from(cfg.fps),
            sim_t * 1e3 / n,
            paint_t * 1e3 / n,
            damaged as f64 * 100.0 / (n * (w * h) as f64),
            rects as f64 / n,
        );
    }
    write_ppm(&path, &px, w, h);
}

/// A white ring with a dark outline, roughly pointer-sized.
fn draw_cursor(px: &mut [u32], w: usize, h: usize, cx: isize, cy: isize, cell: usize) {
    let r = (cell as f32 * 0.45).max(6.0);
    let ri = r.ceil() as isize + 2;
    for y in (cy - ri).max(0)..(cy + ri).min(h as isize) {
        for x in (cx - ri).max(0)..(cx + ri).min(w as isize) {
            let d = (((x - cx).pow(2) + (y - cy).pow(2)) as f32).sqrt();
            let p = &mut px[y as usize * w + x as usize];
            if (d - r).abs() < 1.5 {
                *p = 0xFFFF_FFFF;
            } else if (d - r).abs() < 2.5 {
                *p = 0xFF00_0000;
            }
        }
    }
}

fn bytes_mut(px: &mut [u32]) -> &mut [u8] {
    // SAFETY: plain-old-data view of an aligned u32 buffer.
    unsafe { std::slice::from_raw_parts_mut(px.as_mut_ptr().cast(), px.len() * 4) }
}

fn write_ppm(path: &str, px: &[u32], w: usize, h: usize) {
    let mut out = Vec::with_capacity(w * h * 3 + 32);
    write!(out, "P6\n{w} {h}\n255\n").unwrap();
    for &p in px {
        out.extend_from_slice(&[(p >> 16) as u8, (p >> 8) as u8, p as u8]);
    }
    std::fs::write(path, out).unwrap_or_else(|e| die(&format!("{path}: {e}")));
}
