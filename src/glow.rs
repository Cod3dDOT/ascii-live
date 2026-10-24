//! Soft light behind glyph clusters (`--glow`), computed at cell resolution
//! rather than per pixel: each cell emits its sprite's mean light, a small
//! separable Gaussian spreads it over neighbouring cells, and the result is
//! sampled at cell corners. A pixel gets the bilinear blend of its cell's
//! four corners, ordered-dithered (the glow is a few levels over near-black,
//! where plain rounding bands), added on top of the sprite.
//!
//! Incremental: only cells whose keys changed are re-blurred, corners are
//! quantized, and a cell is repainted only when one of its corners changes.
//! The kernel is truncated, so a change never reaches past `RADIUS` cells.

use fearless_simd::{f32x8, i32x8, prelude::*, u8x32, u32x8};
use fearless_simd_macros::simd;

use crate::color::{to_lin, to_srgb_f32};

/// Gaussian sigma and truncation radius, in cells.
const SIGMA: f32 = 1.2;
const RADIUS: usize = 3;
/// Corner precision: 1/Q of an sRGB level (sub-level steps feed the dither).
const Q: f32 = 4.0;
/// `--glow 1` adds this fraction of the blurred cell light.
const GAIN: f32 = 0.1;
/// Cell light is quantized to this many levels of the brightest sprite's:
/// tone steps then rarely change it, so the glow (and the cells it covers)
/// stays put while glyphs only shimmer. Invisible after the blur.
const LIGHT_LEVELS: f32 = 12.0;

/// 4x4 Bayer thresholds in (0, 1), indexed `[y & 3][x & 3]`.
const BAYER: [[f32; 4]; 4] = {
    let m = [[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]];
    let mut t = [[0.0; 4]; 4];
    let mut y = 0;
    while y < 4 {
        let mut x = 0;
        while x < 4 {
            t[y][x] = (m[y][x] as f32 + 0.5) / 16.0;
            x += 1;
        }
        y += 1;
    }
    t
};

/// `BAYER` rows as 8-lane vectors for each start phase: `[y & 3][x & 3]`.
const DITHER: [[[f32; 8]; 4]; 4] = {
    let mut t = [[[0.0; 8]; 4]; 4];
    let mut y = 0;
    while y < 4 {
        let mut p = 0;
        while p < 4 {
            let mut i = 0;
            while i < 8 {
                t[y][p][i] = BAYER[y][(p + i) & 3];
                i += 1;
            }
            p += 1;
        }
        y += 1;
    }
    t
};

pub(crate) struct Glow {
    gain: f32,
    bg_lin: [f32; 3],
    bg: [f32; 3],
    /// Emitted linear light per atlas slot (mean over the sprite), quantized.
    light: Vec<[f32; 3]>,
    /// Light each cell currently emits (as last blurred).
    cur: Vec<[f32; 3]>,
    weights: [f32; 2 * RADIUS + 1],
    cols: usize,
    rows: usize,
    /// Horizontally blurred light per cell, then fully blurred per cell.
    hpass: Vec<[f32; 3]>,
    blur: Vec<[f32; 3]>,
    /// Quantized glow per cell corner ((cols + 1) x (rows + 1)), in 1/Q
    /// sRGB levels above the background.
    corners: Vec<[u16; 3]>,
}

impl Glow {
    /// `light` is the mean linear light of each atlas slot's sprite.
    pub(crate) fn new(
        strength: f32,
        bg: [u8; 3],
        light: Vec<[f32; 3]>,
        cols: usize,
        rows: usize,
    ) -> Self {
        let bg_lin = bg.map(to_lin);
        let light: Vec<[f32; 3]> = light
            .into_iter()
            .map(|l| std::array::from_fn(|k| (l[k] - bg_lin[k]).max(0.0)))
            .collect();
        let peak = light.iter().flatten().fold(1e-6f32, |m, &v| m.max(v));
        let step = peak / LIGHT_LEVELS;
        let light = light
            .into_iter()
            .map(|l| l.map(|v| (v / step).round() * step))
            .collect();
        let mut weights = std::array::from_fn(|i| {
            let d = i as f32 - RADIUS as f32;
            (-d * d / (2.0 * SIGMA * SIGMA)).exp()
        });
        let sum: f32 = weights.iter().sum();
        weights.iter_mut().for_each(|w| *w /= sum);
        Glow {
            gain: strength * GAIN,
            bg_lin,
            bg: bg.map(f32::from),
            light,
            cur: vec![[0.0; 3]; cols * rows],
            weights,
            cols,
            rows,
            hpass: vec![[0.0; 3]; cols * rows],
            blur: vec![[0.0; 3]; cols * rows],
            corners: vec![[0; 3]; (cols + 1) * (rows + 1)],
        }
    }

    /// Re-blurs after the cells in `[x0, x1) x [y0, y1)` changed key
    /// (`keys` holds atlas slots, row stride `kw`). Calls `stale(i, j)` for
    /// every cell with a corner that changed. Each value is recomputed from
    /// scratch, so the result never depends on update order.
    pub(crate) fn update(
        &mut self,
        keys: &[f32],
        kw: usize,
        [x0, y0, x1, y1]: [usize; 4],
        mut stale: impl FnMut(usize, usize),
    ) {
        let (cols, rows, r) = (self.cols, self.rows, RADIUS);
        // Cells whose (quantized) light actually changed.
        let mut lit = [usize::MAX, usize::MAX, 0, 0];
        for j in y0..y1 {
            for i in x0..x1 {
                let k = keys[j * kw + i];
                let l = if k < 0.0 {
                    [0.0; 3]
                } else {
                    self.light[k as usize]
                };
                if self.cur[j * cols + i] != l {
                    self.cur[j * cols + i] = l;
                    lit = [
                        lit[0].min(i),
                        lit[1].min(j),
                        lit[2].max(i + 1),
                        lit[3].max(j + 1),
                    ];
                }
            }
        }
        if lit[2] <= lit[0] {
            return;
        }
        let [x0, y0, x1, y1] = lit;
        let light = |i: usize, j: usize| self.cur[j * cols + i];
        // Horizontal pass: rows that changed, columns within reach.
        let (hx0, hx1) = (x0.saturating_sub(r), (x1 + r).min(cols));
        for j in y0..y1 {
            for i in hx0..hx1 {
                let mut acc = [0.0; 3];
                for (t, w) in self.weights.iter().enumerate() {
                    let Some(x) = (i + t).checked_sub(r).filter(|&x| x < cols) else {
                        continue;
                    };
                    let l = light(x, j);
                    (0..3).for_each(|c| acc[c] += w * l[c]);
                }
                self.hpass[j * cols + i] = acc;
            }
        }
        // Vertical pass.
        let (vy0, vy1) = (y0.saturating_sub(r), (y1 + r).min(rows));
        for j in vy0..vy1 {
            for i in hx0..hx1 {
                let mut acc = [0.0; 3];
                for (t, w) in self.weights.iter().enumerate() {
                    let Some(y) = (j + t).checked_sub(r).filter(|&y| y < rows) else {
                        continue;
                    };
                    let h = self.hpass[y * cols + i];
                    (0..3).for_each(|c| acc[c] += w * h[c]);
                }
                self.blur[j * cols + i] = acc;
            }
        }
        // Corners touching re-blurred cells.
        for cj in vy0..=vy1 {
            for ci in hx0..=hx1 {
                let q = self.corner(ci, cj);
                let slot = &mut self.corners[cj * (cols + 1) + ci];
                if *slot == q {
                    continue;
                }
                *slot = q;
                for j in cj.saturating_sub(1)..(cj + 1).min(rows) {
                    for i in ci.saturating_sub(1)..(ci + 1).min(cols) {
                        stale(i, j);
                    }
                }
            }
        }
    }

    /// Quantized glow at corner (ci, cj): mean of the up to 4 cells there.
    fn corner(&self, ci: usize, cj: usize) -> [u16; 3] {
        let (mut acc, mut n) = ([0.0f32; 3], 0.0);
        for j in cj.saturating_sub(1)..(cj + 1).min(self.rows) {
            for i in ci.saturating_sub(1)..(ci + 1).min(self.cols) {
                let b = self.blur[j * self.cols + i];
                (0..3).for_each(|c| acc[c] += b[c]);
                n += 1.0;
            }
        }
        std::array::from_fn(|c| {
            let lit = to_srgb_f32(self.bg_lin[c] + self.gain * acc[c] / n);
            // Non-negative after the clamp, so +0.5 and truncation round
            // (f32::round is a libm call without SSE4.1).
            (((lit - self.bg[c]) * Q).clamp(0.0, 255.0 * Q) + 0.5) as u16
        })
    }

    /// Glow (sRGB levels) at the corners of cell (i, j): [tl, tr, bl, br].
    pub(crate) fn cell(&self, i: usize, j: usize) -> [[f32; 3]; 4] {
        let at = |ci: usize, cj: usize| {
            self.corners[cj * (self.cols + 1) + ci].map(|v| f32::from(v) / Q)
        };
        [at(i, j), at(i + 1, j), at(i, j + 1), at(i + 1, j + 1)]
    }
}

/// Adds the glow of one pixel row to `src` into `dst`: per channel a linear
/// ramp from `l` (cell's left edge) to `r` (right edge) across a cell `cs`
/// pixels wide, of which this row starts at in-cell column `xi0`. `(x, y)`
/// is the row's first pixel on screen (dither phase).
#[simd]
#[allow(clippy::too_many_arguments)]
pub(crate) fn add_row<S: Simd>(
    simd: S,
    dst: &mut [u32],
    src: &[u32],
    l: [f32; 3],
    r: [f32; 3],
    (xi0, cs): (usize, usize),
    (x, y): (usize, usize),
) {
    let n = dst.len();
    let src = &src[..n];
    let inv = 1.0 / cs as f32;
    let slope: [f32; 3] = std::array::from_fn(|c| (r[c] - l[c]) * inv);
    // Value at the centre of in-cell column 0.
    let base: [f32; 3] = std::array::from_fn(|c| l[c] + slope[c] * 0.5);
    let bayer = &BAYER[y & 3];
    if n < 8 {
        for (k, (d, &s)) in dst.iter_mut().zip(src).enumerate() {
            let xi = (xi0 + k) as f32;
            let th = bayer[(x + k) & 3];
            let g: [u32; 3] =
                std::array::from_fn(|c| (base[c] + slope[c] * xi + th).clamp(0.0, 255.0) as u32);
            *d = sat_add(s, g[0] << 16 | g[1] << 8 | g[2]);
        }
        return;
    }
    let iota = f32x8::from_slice(simd, &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
    let ramp = Ramp {
        base: base.map(|v| f32x8::splat(simd, v)),
        slope: slope.map(|v| f32x8::splat(simd, v)),
        zero: f32x8::splat(simd, 0.0),
        max: f32x8::splat(simd, 255.0),
    };
    // Chunks of 8; the last one overlaps instead of a scalar tail (it is
    // recomputed from `src`, so writing those pixels twice is harmless).
    for k in (0..n).step_by(8) {
        let k = k.min(n - 8);
        // Dither phase of column k is (x + k) & 3.
        let th = f32x8::from_slice(simd, &DITHER[y & 3][(x + k) & 3]);
        let xi = iota + f32x8::splat(simd, (xi0 + k) as f32);
        let g = (ramp.channel(xi, th, 0) << 16)
            | (ramp.channel(xi, th, 1) << 8)
            | ramp.channel(xi, th, 2);
        let s = u32x8::from_slice(simd, &src[k..k + 8]);
        let out: u32x8<S> = s
            .bitcast::<u8x32<S>>()
            .saturating_add(g.bitcast::<u8x32<S>>())
            .bitcast();
        out.store_slice(&mut dst[k..k + 8]);
    }
}

/// Per-channel glow ramp of one row, as vectors. A struct with
/// `#[inline(always)]` methods rather than closures: closures are separate
/// functions that don't inherit the caller's target features, so their SIMD
/// ops would compile to calls.
struct Ramp<S: Simd> {
    base: [f32x8<S>; 3],
    slope: [f32x8<S>; 3],
    zero: f32x8<S>,
    max: f32x8<S>,
}

impl<S: Simd> Ramp<S> {
    /// Dithered glow levels of channel `c` at in-cell columns `xi`.
    #[inline(always)]
    fn channel(&self, xi: f32x8<S>, th: f32x8<S>, c: usize) -> u32x8<S> {
        // Plain mul + add (no FMA) so this matches the scalar path bit for bit.
        let v = xi * self.slope[c] + self.base[c] + th;
        // Via i32: native on x86 (u32 isn't) and exact for 0..=255.
        i32x8::truncate_from(v.max(self.zero).min(self.max)).bitcast()
    }
}

#[inline(always)]
fn sat_add(a: u32, b: u32) -> u32 {
    let ch = |s: u32| (((a >> s) & 0xFF) + ((b >> s) & 0xFF)).min(255) << s;
    (a & 0xFF00_0000) | ch(16) | ch(8) | ch(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn glow(cols: usize, rows: usize) -> Glow {
        // Slot 0: dark (no light), slot 1: bright.
        Glow::new(1.0, [4, 3, 2], vec![[0.0; 3], [0.5, 0.4, 0.2]], cols, rows)
    }

    #[test]
    fn light_spreads_locally_and_symmetrically() {
        let (cols, rows) = (20, 12);
        let mut keys = vec![0.0; cols * rows];
        keys[6 * cols + 10] = 1.0;
        let mut g = glow(cols, rows);
        let mut stale = Vec::new();
        g.update(&keys, cols, [0, 0, cols, rows], |i, j| stale.push((i, j)));
        let at = |ci: usize, cj: usize| g.corners[cj * (cols + 1) + ci][0];
        // Brightest around the lit cell, symmetric, zero past the radius.
        assert!(at(10, 6) > 0 && at(10, 6) == at(11, 7) && at(10, 7) == at(11, 6));
        assert!(at(10, 6) > at(9, 6) && at(9, 6) > at(8, 6));
        assert_eq!(at(10 - RADIUS - 1, 6), 0);
        assert!(
            stale
                .iter()
                .all(|&(i, j)| i.abs_diff(10) <= RADIUS + 1 && j.abs_diff(6) <= RADIUS + 1)
        );
    }

    #[test]
    fn incremental_update_matches_full_recompute() {
        let (cols, rows) = (24, 16);
        let mut keys: Vec<f32> = (0..cols * rows)
            .map(|i| f32::from(u8::from((i * 7) % 5 == 0)))
            .collect();
        let mut inc = glow(cols, rows);
        inc.update(&keys, cols, [0, 0, cols, rows], |_, _| {});
        for (i, j) in [(3, 4), (20, 15), (0, 0), (12, 8)] {
            keys[j * cols + i] = 1.0 - keys[j * cols + i];
            let mut stale = Vec::new();
            inc.update(&keys, cols, [i, j, i + 1, j + 1], |a, b| stale.push((a, b)));
            let mut full = glow(cols, rows);
            full.update(&keys, cols, [0, 0, cols, rows], |_, _| {});
            assert!(inc.corners == full.corners, "after flipping ({i}, {j})");
            assert!(!stale.is_empty());
        }
    }

    #[test]
    fn simd_row_matches_scalar_and_dither_averages_out() {
        let simd = fearless_simd::Level::new();
        fearless_simd::dispatch!(simd, s => {
            let src = vec![0xFF04_0302u32; 37];
            let (l, r) = ([2.3, 1.1, 0.0], [5.6, 2.2, 0.4]);
            let mut a = vec![0; 37];
            add_row(s, &mut a, &src, l, r, (3, 40), (5, 9));
            for k in 0..37 {
                let mut b = [0u32; 1];
                add_row(s, &mut b, &src[k..k + 1], l, r, (3 + k, 40), (5 + k, 9));
                assert_eq!(a[k], b[0], "pixel {k}");
            }
            // Dithered red over a flat 2.5-level glow averages to ~2.5.
            let mut flat = vec![0; 64];
            let mut sum = 0;
            for y in 0..4 {
                add_row(s, &mut flat, &vec![0xFF00_0000; 64], [2.5; 3], [2.5; 3], (0, 64), (0, y));
                sum += flat.iter().map(|p| (p >> 16) & 0xFF).sum::<u32>();
            }
            let mean = sum as f32 / 256.0;
            assert!((mean - 2.5).abs() < 0.1, "{mean}");
        });
    }
}
