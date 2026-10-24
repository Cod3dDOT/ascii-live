//! Antialiased glyph sprites rasterized from signed distance functions,
//! so they stay crisp at any cell size. Within `[g, g+1)` the last `MORPH`
//! fraction blends glyph `g`'s SDF toward `g+1`; morph frames are baked into
//! the atlas, so a shape change is still one blit.

/// Fraction of each glyph's value range spent morphing into the next glyph.
/// The rest field avoids this band, so the settled image shows clean glyphs.
pub const MORPH: f32 = 0.4;

fn sd_box(px: f32, py: f32, bx: f32, by: f32, r: f32) -> f32 {
    let qx = px.abs() - bx + r;
    let qy = py.abs() - by + r;
    let ox = qx.max(0.0);
    let oy = qy.max(0.0);
    (ox * ox + oy * oy).sqrt() + qx.max(qy).min(0.0) - r
}

/// Distance to the segment `a`-`b`, minus `r`: a stroke with round caps.
fn sd_seg(x: f32, y: f32, (ax, ay): (f32, f32), (bx, by): (f32, f32), r: f32) -> f32 {
    let (px, py, dx, dy) = (x - ax, y - ay, bx - ax, by - ay);
    let h = ((px * dx + py * dy) / (dx * dx + dy * dy)).clamp(0.0, 1.0);
    (px - dx * h).hypot(py - dy * h) - r
}

fn sd_ring(x: f32, y: f32, radius: f32, half_width: f32) -> f32 {
    (x.hypot(y) - radius).abs() - half_width
}

/// Characters with a shape, one per look (aliases aside), for messages.
pub const SUPPORTED: &str = ". - + | = : # o x / \\ ~ ^ * @ %";

/// Whether `c` has a shape (see [`SUPPORTED`]); `' '` is the blank.
pub fn supported(c: char) -> bool {
    shape(c, 0.0, 0.0).is_some()
}

/// Signed distance in cell units (cell is the unit square centered at 0,
/// y pointing down), or `None` for a character without a shape. Shapes
/// stay within about 0.4 of the center so they never touch the cell edge.
fn shape(glyph: char, x: f32, y: f32) -> Option<f32> {
    Some(match glyph {
        ' ' => f32::INFINITY,
        '-' => sd_box(x, y, 0.32, 0.13, 0.01),
        '+' => sd_box(x, y, 0.375, 0.125, 0.0).min(sd_box(x, y, 0.125, 0.375, 0.0)),
        '|' => sd_box(x, y, 0.13, 0.32, 0.01),
        '=' => sd_box(x, y - 0.12, 0.28, 0.06, 0.01).min(sd_box(x, y + 0.12, 0.28, 0.06, 0.01)),
        ':' => ((x * x + (y - 0.17).powi(2)).sqrt() - 0.075)
            .min((x * x + (y + 0.17).powi(2)).sqrt() - 0.075),
        '#' | '█' => sd_box(x, y, 0.36, 0.36, 0.04),
        'o' | 'O' | '0' => sd_ring(x, y, 0.297, 0.078),
        'x' | 'X' => {
            let s = std::f32::consts::FRAC_1_SQRT_2;
            let (u, v) = ((x + y) * s, (x - y) * s);
            sd_box(u, v, 0.49, 0.094, 0.0).min(sd_box(v, u, 0.49, 0.094, 0.0))
        }
        '/' => sd_seg(x, y, (-0.24, 0.3), (0.24, -0.3), 0.09),
        '\\' => sd_seg(x, y, (-0.24, -0.3), (0.24, 0.3), 0.09),
        '^' => sd_seg(x, y, (-0.27, 0.12), (0.0, -0.2), 0.085).min(sd_seg(
            x,
            y,
            (0.0, -0.2),
            (0.27, 0.12),
            0.085,
        )),
        '~' => {
            // One sine period; distance to the curve is |y - f(x)| / |grad|.
            let (a, k) = (0.09, std::f32::consts::TAU / 0.6);
            let wave = (y - a * (k * x).sin()).abs() / (a * k * (k * x).cos()).hypot(1.0);
            (wave - 0.08).max(x.abs() - 0.3)
        }
        '*' => (0..3)
            .map(|i| {
                let (s, c) = (std::f32::consts::FRAC_PI_3 * i as f32).sin_cos();
                let (dx, dy) = (s * 0.3, -c * 0.3); // spoke pairs from vertical
                sd_seg(x, y, (-dx, -dy), (dx, dy), 0.075)
            })
            .fold(f32::INFINITY, f32::min),
        '@' => sd_ring(x, y, 0.33, 0.055)
            .min(sd_ring(x, y, 0.12, 0.06))
            .min(sd_seg(x, y, (0.12, -0.12), (0.12, 0.16), 0.06))
            .min(sd_seg(x, y, (0.12, 0.16), (0.27, 0.19), 0.06)),
        '%' => sd_seg(x, y, (-0.24, 0.3), (0.24, -0.3), 0.06)
            .min((x + 0.2).hypot(y + 0.2) - 0.09)
            .min((x - 0.2).hypot(y - 0.2) - 0.09),
        '.' => (x * x + y * y).sqrt() - 0.175,
        _ => return None,
    })
}

/// [`shape`], with unknown characters drawn as a dot (the CLI rejects them;
/// this keeps library callers from crashing).
fn sdf(glyph: char, x: f32, y: f32) -> f32 {
    shape(glyph, x, y).unwrap_or_else(|| shape('.', x, y).unwrap())
}

/// Glyph index and morph progress toward the next glyph for value `v`.
pub fn shape_at(v: f32, glyphs: usize) -> (usize, f32) {
    let g = (v.max(0.0) as usize).min(glyphs - 1);
    if g + 1 >= glyphs {
        return (g, 0.0);
    }
    let t = ((v - g as f32 - (1.0 - MORPH)) / MORPH).clamp(0.0, 1.0);
    (g, t * t * (3.0 - 2.0 * t))
}

/// Distance field of glyph `a` morphed toward `b` by `t`.
fn sdf_morph(a: char, b: char, t: f32, x: f32, y: f32) -> f32 {
    if t <= 0.0 {
        return sdf(a, x, y);
    }
    if a == ' ' {
        // Grow `b` out of a point (keeps the stroke shape, scales it).
        let s = t.max(1e-3);
        return sdf(b, x / s, y / s) * s;
    }
    // Blend outlines; the bump keeps strokes solid halfway through, where a
    // plain lerp of two disjoint shapes thins both out.
    let d = sdf(a, x, y) + (sdf(b, x, y) - sdf(a, x, y)) * t;
    d - 0.04 * t * (1.0 - t)
}

/// Alpha mask (`cs` x `cs`) for glyph `g` of `ramp` morphed `t` toward `g+1`.
pub fn mask(ramp: &[char], g: usize, t: f32, cs: usize) -> Vec<u8> {
    let (a, b) = (ramp[g], ramp.get(g + 1).copied().unwrap_or(' '));
    let s = cs as f32;
    let mut m = vec![0u8; cs * cs];
    if a == ' ' && t <= 0.0 {
        return m;
    }
    for y in 0..cs {
        for x in 0..cs {
            let fx = (x as f32 + 0.5) / s - 0.5;
            let fy = (y as f32 + 0.5) / s - 0.5;
            let d = sdf_morph(a, b, t, fx, fy) * s; // pixels
            m[y * cs + x] = ((0.5 - d).clamp(0.0, 1.0) * 255.0).round() as u8;
        }
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    const RAMP: [char; 18] = [
        ' ', '.', '-', '+', 'o', 'x', '|', '=', ':', '#', '/', '\\', '~', '^', '*', '@', '%', '?',
    ];

    /// No glyph (or morph frame) may bleed into the neighbouring cell, or
    /// incremental cell blits would leave stale pixels behind.
    #[test]
    fn glyphs_stay_inside_their_cell() {
        for cs in [8, 21, 31, 64] {
            for (g, c) in RAMP.iter().enumerate() {
                for t in [0.0, 0.25, 0.5, 0.75, 1.0] {
                    let m = mask(&RAMP, g, t, cs);
                    for i in 0..cs {
                        for (x, y) in [(i, 0), (i, cs - 1), (0, i), (cs - 1, i)] {
                            assert!(m[y * cs + x] < 128, "{c:?} t={t} cs={cs} touches edge");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn blank_is_empty_and_glyphs_are_not() {
        assert!(mask(&RAMP, 0, 0.0, 32).iter().all(|&a| a == 0));
        for (g, c) in RAMP.iter().enumerate().skip(1) {
            assert!(mask(&RAMP, g, 0.0, 32).contains(&255), "{c:?}");
        }
    }

    #[test]
    fn supported_glyphs() {
        for c in SUPPORTED.split(' ').chain([" ", "O", "0", "X", "█"]) {
            let c = c.chars().next().unwrap_or(' ');
            assert!(supported(c), "{c:?}");
        }
        for c in ['?', 'a', '&', '\t'] {
            assert!(!supported(c), "{c:?}");
        }
    }

    #[test]
    fn shape_at_morph_band() {
        assert_eq!(shape_at(1.0, 6), (1, 0.0));
        assert_eq!(shape_at(1.0 + (1.0 - MORPH) * 0.99, 6).1, 0.0);
        assert!((shape_at(1.999, 6).1 - 1.0).abs() < 1e-2);
        assert_eq!(shape_at(5.7, 6), (5, 0.0)); // last glyph never morphs
        assert_eq!(shape_at(-1.0, 6), (0, 0.0));
    }
}
