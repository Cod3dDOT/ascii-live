//! Live ASCII fluid wallpaper for wlr-layer-shell compositors.
//!
//! [`field`] builds the rest image, [`sim`] advects it (sparse stable fluids),
//! [`render`] blits [`glyph`] sprites for changed cells only, and [`app`]
//! drives this per output.

pub mod app;
pub mod cli;
pub mod color;
pub mod field;
pub mod glyph;
pub mod instance;
pub mod pacing;
pub mod render;
pub mod sim;
pub mod workload;

pub struct Config {
    pub style: field::Style,
    pub params: sim::Params,
    pub fps: f32,
    /// false = `--static`: draw once, take no input, free everything.
    pub interactive: bool,
}
