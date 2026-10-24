//! Live ASCII fluid wallpaper for wlr-layer-shell compositors.
//!
//! [`field`] builds the rest image, [`sim`] advects it (sparse stable fluids),
//! [`render`] blits [`glyph`] sprites for changed cells only, and [`app`]
//! drives this per output.

pub mod app;
pub mod cli;
pub mod color;
pub mod field;
pub mod glow;
pub mod glyph;
pub mod instance;
pub mod pacing;
pub mod render;
pub mod sim;
pub mod touch;
pub mod workload;

pub struct Config {
    pub style: field::Style,
    pub params: sim::Params,
    /// Frame cap while animating; 0 = the output's refresh rate.
    pub fps: f32,
    /// false = `--static`: draw once, take no input, free everything.
    pub interactive: bool,
}

impl Config {
    pub fn rate(&self) -> pacing::Rate {
        if self.fps > 0.0 {
            pacing::Rate::Fps(self.fps)
        } else {
            pacing::Rate::Display
        }
    }
}
