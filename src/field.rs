//! Procedural wallpaper: a domain-warped fBm field quantized per cell onto a
//! glyph ramp (e.g. ` .-+ox`). A cell value `v` lies in `[0, ramp.len())`:
//! `floor(v)` picks the glyph and `fract(v)` its brightness, so advecting `v`
//! morphs glyphs along the ramp.

use crate::color::Palette;
use crate::glyph::MORPH;

pub struct Style {
    /// Glyph ramp, darkest first; index 0 is always drawn as blank.
    pub ramp: Vec<char>,
    pub palette: Palette,
    /// Color curve exponent along the ramp (>1 keeps more cells dim).
    pub gamma: f32,
    /// Cell rows per screen height, so the look is identical on every output
    /// regardless of resolution or scale.
    pub rows: f32,
    /// Noise feature size, in cells.
    pub scale: f32,
    pub octaves: u32,
    /// Domain warp amount (0 = blobs, ~2 = marbled smoke).
    pub warp: f32,
    /// Fraction of cells left blank, 0..1.
    pub blank: f32,
    /// Contrast of the field before quantization.
    pub contrast: f32,
    pub seed: u32,
    /// One field across all outputs, placed by their desktop layout, with
    /// one cell size (from the tallest output).
    pub span: bool,
    /// Light behind glyph clusters, 0 = off (see [`crate::glow`]).
    pub glow: f32,
}

impl Style {
    pub fn glyphs(&self) -> usize {
        self.ramp.len()
    }

    /// Cell edge in buffer pixels for a buffer `height` pixels tall.
    pub fn cell_px(&self, height: usize) -> usize {
        ((height as f32 / self.rows.max(1.0)).round() as usize).max(4)
    }

    /// Color for glyph `g` (>= 1) at brightness `t` in 0..1.
    pub fn color(&self, g: usize, t: f32) -> [u8; 3] {
        let n = (self.glyphs() - 1).max(1) as f32;
        self.palette
            .at(((g as f32 - 1.0 + t) / n).clamp(0.0, 1.0).powf(self.gamma))
    }

    /// Rest field for a `cols` x `rows` cell grid.
    pub fn generate(&self, cols: usize, rows: usize) -> Vec<f32> {
        let mut raw = self.raw((0, 0), cols, rows);
        let levels = self.levels(&raw);
        self.quantize(&mut raw, levels);
        raw
    }

    /// Rest field for a `cols` x `rows` window whose first cell is cell
    /// `origin` of one desktop-wide grid. Levels come from a fixed sample of
    /// that grid, so adjacent windows join seamlessly.
    pub fn generate_spanned(&self, origin: (i32, i32), cols: usize, rows: usize) -> Vec<f32> {
        let levels = self.levels(&self.raw((0, 0), SPAN_SAMPLE.0, SPAN_SAMPLE.1));
        let mut raw = self.raw(origin, cols, rows);
        self.quantize(&mut raw, levels);
        raw
    }

    /// Unquantized fBm over a window of the cell grid.
    fn raw(&self, (gx, gy): (i32, i32), cols: usize, rows: usize) -> Vec<f32> {
        let s = 1.0 / self.scale.max(0.5);
        (0..rows * cols)
            .map(|i| {
                let x = (gx as f32 + (i % cols) as f32) * s;
                let y = (gy as f32 + (i / cols) as f32) * s;
                let (wx, wy) = if self.warp > 0.0 {
                    (
                        fbm(self.seed ^ 0x51, x + 1.7, y + 9.2, 3),
                        fbm(self.seed ^ 0xA7, x + 8.3, y + 2.8, 3),
                    )
                } else {
                    (0.0, 0.0)
                };
                fbm(
                    self.seed,
                    x + self.warp * wx,
                    y + self.warp * wy,
                    self.octaves,
                )
            })
            .collect()
    }

    /// (blank threshold, bright end) of `raw`, by rank, so `blank` is an
    /// exact coverage fraction independent of the noise distribution.
    fn levels(&self, raw: &[f32]) -> (f32, f32) {
        let mut sorted = raw.to_vec();
        sorted.sort_unstable_by(f32::total_cmp);
        let q = |f: f32| sorted[((sorted.len() - 1) as f32 * f.clamp(0.0, 1.0)) as usize];
        let lo = q(self.blank);
        (lo, q(0.995).max(lo + 1e-6))
    }

    /// Maps raw noise onto the ramp: blank below `lo`, contrast on the rest.
    fn quantize(&self, raw: &mut [f32], (lo, hi): (f32, f32)) {
        let n = self.glyphs() as f32;
        for v in raw {
            if *v <= lo {
                *v = 0.0;
                continue;
            }
            let t = ((*v - lo) / (hi - lo)).clamp(0.0, 1.0);
            let t = (t.powf(1.0 / self.contrast.max(0.1))).min(0.9999);
            let x = 1.0 + t * (n - 1.0);
            // Keep resting cells out of the morph band (clean glyphs at rest).
            let g = x.floor();
            *v = if g < n - 1.0 {
                g + (x - g) * (1.0 - MORPH)
            } else {
                x
            };
        }
    }
}

/// Cells sampled for `--span` levels: a fixed window, so every output
/// agrees (fBm is stationary, so any large window is representative).
const SPAN_SAMPLE: (usize, usize) = (256, 144);

#[inline]
fn hash(seed: u32, x: i32, y: i32) -> u32 {
    let mut h = seed ^ (x as u32).wrapping_mul(0x27d4_eb2d) ^ (y as u32).wrapping_mul(0x1656_67b1);
    h ^= h >> 15;
    h = h.wrapping_mul(0x85eb_ca6b);
    h ^= h >> 13;
    h = h.wrapping_mul(0xc2b2_ae35);
    h ^ (h >> 16)
}

const S2: f32 = std::f32::consts::FRAC_1_SQRT_2;

/// 16 unit gradient directions (k * 22.5°): no trig in the noise loop.
const GRADS: [(f32, f32); 16] = [
    (1.0000000, 0.0000000),
    (0.9238795, 0.3826834),
    (S2, S2),
    (0.3826834, 0.9238795),
    (0.0000000, 1.0000000),
    (-0.3826834, 0.9238795),
    (-S2, S2),
    (-0.9238795, 0.3826834),
    (-1.0000000, 0.0000000),
    (-0.9238795, -0.3826834),
    (-S2, -S2),
    (-0.3826834, -0.9238795),
    (-0.0000000, -1.0000000),
    (0.3826834, -0.9238795),
    (S2, -S2),
    (0.9238795, -0.3826834),
];

/// Gradient (Perlin-style) noise in roughly -1..1.
fn noise(seed: u32, x: f32, y: f32) -> f32 {
    let (xf, yf) = (x.floor(), y.floor());
    let (xi, yi) = (xf as i32, yf as i32);
    let (fx, fy) = (x - xf, y - yf);
    let grad = |ix: i32, iy: i32, dx: f32, dy: f32| {
        let (gx, gy) = GRADS[(hash(seed, ix, iy) >> 7) as usize & 15];
        gx * dx + gy * dy
    };
    let fade = |t: f32| t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
    let (u, v) = (fade(fx), fade(fy));
    let a = grad(xi, yi, fx, fy);
    let b = grad(xi + 1, yi, fx - 1.0, fy);
    let c = grad(xi, yi + 1, fx, fy - 1.0);
    let d = grad(xi + 1, yi + 1, fx - 1.0, fy - 1.0);
    let ab = a + (b - a) * u;
    let cd = c + (d - c) * u;
    (ab + (cd - ab) * v) * 1.414
}

fn fbm(seed: u32, x: f32, y: f32, octaves: u32) -> f32 {
    let (mut sum, mut amp, mut freq, mut norm) = (0.0, 1.0, 1.0, 0.0);
    for o in 0..octaves.max(1) {
        sum += amp * noise(seed.wrapping_add(o * 1013), x * freq, y * freq);
        norm += amp;
        amp *= 0.5;
        freq *= 2.03;
    }
    sum / norm
}

#[cfg(test)]
mod tests {
    use super::*;

    fn style(blank: f32) -> Style {
        Style {
            ramp: " .-+ox".chars().collect(),
            palette: Palette::new([92, 169, 183], None),
            gamma: 1.2,
            rows: 54.0,
            scale: 14.0,
            octaves: 4,
            warp: 0.8,
            blank,
            contrast: 1.3,
            seed: 7,
            span: false,
            glow: 0.0,
        }
    }

    #[test]
    fn blank_fraction_is_exact_and_values_avoid_morph_band() {
        let (cols, rows) = (96, 54);
        for blank in [0.0, 0.3, 0.45, 0.9] {
            let f = style(blank).generate(cols, rows);
            let zeros = f.iter().filter(|&&v| v == 0.0).count() as f32 / f.len() as f32;
            assert!((zeros - blank).abs() < 0.01, "blank {blank}: got {zeros}");
            for &v in &f {
                assert!((0.0..6.0).contains(&v));
                if v > 0.0 && v < 5.0 {
                    assert!(
                        v.fract() <= 1.0 - MORPH + 1e-4,
                        "{v} rests in the morph band"
                    );
                }
            }
        }
    }

    #[test]
    fn cell_px_scales_with_height() {
        let s = style(0.45);
        assert_eq!(s.cell_px(1080), 20);
        assert_eq!(s.cell_px(2160), 40);
        assert_eq!(s.cell_px(10), 4);
    }

    #[test]
    fn spanned_windows_join_seamlessly() {
        let s = style(0.45);
        let whole = s.generate_spanned((-10, 5), 50, 20);
        let right = s.generate_spanned((15, 5), 25, 20);
        for y in 0..20 {
            assert_eq!(whole[y * 50 + 25..y * 50 + 50], right[y * 25..y * 25 + 25]);
        }
    }

    #[test]
    fn spanned_levels_match_the_requested_coverage() {
        let f = style(0.45).generate_spanned((300, -40), 200, 100);
        let zeros = f.iter().filter(|&&v| v == 0.0).count() as f32 / f.len() as f32;
        assert!((zeros - 0.45).abs() < 0.08, "{zeros}");
    }

    #[test]
    fn seed_changes_field() {
        let mut a = style(0.45);
        let b = a.generate(64, 36);
        a.seed = 8;
        assert_ne!(b, a.generate(64, 36));
    }
}
