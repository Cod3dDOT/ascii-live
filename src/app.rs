//! Wayland layer-shell front end: one background surface per output.
//! Frames are capped to `fps` and gated on frame callbacks, so hidden outputs
//! stop ticking; once the fluid settles the process sleeps until input.
//! Static mode frees sim, renderer and buffers after the first commit (the
//! compositor keeps its own copy).

use std::time::{Duration, Instant};

use smithay_client_toolkit::{
    compositor::{CompositorHandler, CompositorState, Region},
    delegate_compositor, delegate_layer, delegate_output, delegate_pointer, delegate_registry,
    delegate_seat, delegate_shm,
    output::{OutputHandler, OutputState},
    reexports::{
        calloop::{
            EventLoop, LoopHandle, RegistrationToken,
            timer::{TimeoutAction, Timer},
        },
        calloop_wayland_source::WaylandSource,
        protocols::wp::{
            cursor_shape::v1::client::wp_cursor_shape_device_v1::{Shape, WpCursorShapeDeviceV1},
            fractional_scale::v1::client::{
                wp_fractional_scale_manager_v1::WpFractionalScaleManagerV1,
                wp_fractional_scale_v1::{self, WpFractionalScaleV1},
            },
            viewporter::client::{wp_viewport::WpViewport, wp_viewporter::WpViewporter},
        },
    },
    registry::{ProvidesRegistryState, RegistryState},
    registry_handlers,
    seat::{
        Capability, SeatHandler, SeatState,
        pointer::{
            PointerEvent, PointerEventKind, PointerHandler, cursor_shape::CursorShapeManager,
        },
    },
    shell::{
        WaylandSurface,
        wlr_layer::{
            Anchor, KeyboardInteractivity, Layer, LayerShell, LayerShellHandler, LayerSurface,
            LayerSurfaceConfigure,
        },
    },
    shm::{
        Shm, ShmHandler,
        slot::{Buffer, SlotPool},
    },
};
use wayland_client::{
    Connection, Dispatch, Proxy, QueueHandle,
    backend::ObjectId,
    globals::registry_queue_init,
    protocol::{wl_output, wl_pointer, wl_seat, wl_shm, wl_surface},
};

use crate::{
    Config,
    pacing::{Pacer, Present, RETRY},
    render::{Layout, Renderer},
    sim::Sim,
};

const BTN_LEFT: u32 = 0x110;
const BTN_RIGHT: u32 = 0x111;
const BTN_MIDDLE: u32 = 0x112;

/// Logs an shm error once per outage (until the next successful present).
fn shm_failed(ok: &mut bool, what: &str, e: impl std::fmt::Display) -> Present {
    if std::mem::take(ok) {
        eprintln!("situ: {what}: {e} (retrying every {}s)", RETRY.as_secs());
    }
    Present::Failed
}

struct Buf {
    buffer: Buffer,
    /// Renderer-side state index for this buffer.
    state: usize,
}

struct View {
    renderer: Renderer,
    sim: Sim,
    bufs: Vec<Buf>,
}

struct Out {
    output: wl_output::WlOutput,
    layer: LayerSurface,
    /// Buffer pixels per logical pixel (fractional with wp_fractional_scale).
    scale: f64,
    /// Present when the compositor supports fractional scaling.
    frac: Option<(WpFractionalScaleV1, WpViewport)>,
    size: (u32, u32),
    view: Option<View>,
    pacer: Pacer,
    timer: Option<RegistrationToken>,
    cursor: Option<(f32, f32)>,
}

pub struct App {
    registry_state: RegistryState,
    seat_state: SeatState,
    output_state: OutputState,
    compositor: CompositorState,
    layer_shell: LayerShell,
    shm: Shm,
    /// Created on demand; dropped in static mode once every output is drawn.
    pool: Option<SlotPool>,
    /// False while presents fail, so each outage is logged once.
    shm_ok: bool,
    cursor_shape: Option<CursorShapeManager>,
    /// Fractional scaling needs both globals.
    fractional: Option<(WpFractionalScaleManagerV1, WpViewporter)>,
    cursor_device: Option<WpCursorShapeDeviceV1>,
    pointer: Option<wl_pointer::WlPointer>,
    outs: Vec<Out>,
    cfg: Config,
    qh: QueueHandle<App>,
    handle: LoopHandle<'static, App>,
}

/// Only returns on error.
pub fn run(cfg: Config) -> Result<!, Box<dyn std::error::Error>> {
    let conn = Connection::connect_to_env()?;
    let (globals, queue) = registry_queue_init::<App>(&conn)?;
    let qh = queue.handle();
    let mut event_loop: EventLoop<App> = EventLoop::try_new()?;
    WaylandSource::new(conn, queue)
        .insert(event_loop.handle())
        .map_err(|e| e.error)?;

    let shm = Shm::bind(&globals, &qh)?;
    let mut app = App {
        registry_state: RegistryState::new(&globals),
        seat_state: SeatState::new(&globals, &qh),
        output_state: OutputState::new(&globals, &qh),
        compositor: CompositorState::bind(&globals, &qh)?,
        layer_shell: LayerShell::bind(&globals, &qh)
            .map_err(|_| "compositor lacks wlr-layer-shell")?,
        shm,
        pool: None,
        shm_ok: true,
        cursor_shape: CursorShapeManager::bind(&globals, &qh).ok(),
        fractional: globals
            .bind(&qh, 1..=1, ())
            .ok()
            .zip(globals.bind(&qh, 1..=1, ()).ok()),
        cursor_device: None,
        pointer: None,
        outs: Vec::new(),
        cfg,
        qh,
        handle: event_loop.handle(),
    };
    loop {
        // Blocks in epoll until Wayland events or a pending frame timer.
        event_loop.dispatch(None, &mut app)?;
    }
}

impl App {
    fn set_scale(&mut self, i: usize, scale: f64) {
        if self.outs[i].scale != scale {
            self.outs[i].scale = scale;
            self.rebuild(i);
        }
    }

    fn interval(&self) -> Duration {
        Duration::from_secs_f32(1.0 / self.cfg.fps.max(1.0))
    }

    fn remove(&mut self, i: usize) {
        let out = self.outs.remove(i);
        if let Some(t) = out.timer {
            self.handle.remove(t);
        }
        if let Some((f, v)) = out.frac {
            f.destroy();
            v.destroy();
        }
    }

    fn find(&self, surface: &wl_surface::WlSurface) -> Option<usize> {
        self.outs
            .iter()
            .position(|o| o.layer.wl_surface() == surface)
    }

    fn rebuild(&mut self, i: usize) {
        let out = &mut self.outs[i];
        let px = |v: u32| (f64::from(v) * out.scale).round() as usize;
        let (w, h) = (px(out.size.0), px(out.size.1));
        if w == 0 || h == 0 {
            return;
        }
        let layout = Layout::new(w, h, self.cfg.style.cell_px(h));
        let rest = self.cfg.style.generate(layout.cols, layout.rows);
        let sim = Sim::new(layout.cols, layout.rows, rest, self.cfg.params.clone());
        out.view = Some(View {
            renderer: Renderer::new(layout, &self.cfg.style),
            sim,
            bufs: Vec::new(),
        });
        match &out.frac {
            Some((_, viewport)) => viewport.set_destination(out.size.0 as i32, out.size.1 as i32),
            None => out.layer.wl_surface().set_buffer_scale(out.scale as i32),
        }
        // Nothing else retries a failed first frame: the output would stay
        // blank until pointer input.
        if self.present(i) == Present::Failed {
            let wait = self.outs[i].pacer.retry();
            self.arm(i, wait);
        }
    }

    /// Paints changed cells into a free buffer and commits.
    fn present(&mut self, i: usize) -> Present {
        let out = &mut self.outs[i];
        let Some(view) = out.view.as_mut() else {
            return Present::Skipped;
        };
        view.renderer.sync(&mut view.sim);
        if !view.renderer.has_damage() {
            return Present::Skipped;
        }
        let l = &view.renderer.layout;
        let (w, h) = (l.width as i32, l.height as i32);
        let pool = match &mut self.pool {
            Some(p) => p,
            None => match SlotPool::new((w * h * 4) as usize, &self.shm) {
                Ok(p) => self.pool.insert(p),
                Err(e) => return shm_failed(&mut self.shm_ok, "shm pool", e),
            },
        };
        let free = view
            .bufs
            .iter()
            .position(|b| b.buffer.canvas(pool).is_some());
        let idx = match free {
            Some(idx) => idx,
            None if view.bufs.len() < 3 => {
                let buffer = match pool.create_buffer(w, h, w * 4, wl_shm::Format::Xrgb8888) {
                    Ok((buffer, _)) => buffer,
                    Err(e) => return shm_failed(&mut self.shm_ok, "shm buffer", e),
                };
                let state = view.renderer.add_buffer();
                view.bufs.push(Buf { buffer, state });
                view.bufs.len() - 1
            }
            // Compositor holds every buffer; retry on the next tick.
            None => return Present::Busy,
        };
        let buf = &view.bufs[idx];
        let Some(canvas) = buf.buffer.canvas(pool) else {
            return Present::Busy;
        };
        view.renderer
            .paint(view.sim.level(), buf.state, canvas, (w * 4) as usize);

        let surface = out.layer.wl_surface();
        // Attach before taking damage, so a failure keeps it for the retry.
        if let Err(e) = buf.buffer.attach_to(surface) {
            return shm_failed(&mut self.shm_ok, "attach buffer", e);
        }
        for (x, y, rw, rh) in view.renderer.take_damage() {
            surface.damage_buffer(x, y, rw, rh);
        }
        self.shm_ok = true;
        if !self.cfg.interactive {
            out.layer.commit();
            // The compositor has (or will copy) the pixels; free everything.
            out.view = None;
            if self.outs.iter().all(|o| o.view.is_none()) {
                self.pool = None;
            }
            return Present::Committed;
        }
        surface.frame(&self.qh, surface.clone());
        out.pacer.frame_requested();
        out.layer.commit();
        Present::Committed
    }

    /// Arms the tick timer the pacer asked for.
    fn arm(&mut self, i: usize, wait: Option<Duration>) {
        let Some(wait) = wait else { return };
        let out = &mut self.outs[i];
        if out.view.is_none() {
            return out.pacer.disarm();
        }
        let id = out.layer.wl_surface().id();
        let token = self
            .handle
            .insert_source(Timer::from_duration(wait), move |_, _, app| {
                app.on_timer(&id);
                TimeoutAction::Drop
            });
        match token {
            Ok(t) => out.timer = Some(t),
            Err(_) => out.pacer.disarm(),
        }
    }

    fn on_timer(&mut self, id: &ObjectId) {
        let Some(i) = self
            .outs
            .iter()
            .position(|o| &o.layer.wl_surface().id() == id)
        else {
            return;
        };
        self.outs[i].timer = None;
        self.tick(i);
    }

    fn tick(&mut self, i: usize) {
        let out = &mut self.outs[i];
        let dt = out.pacer.tick(Instant::now());
        let Some(view) = out.view.as_mut() else {
            return;
        };
        view.sim.advance(dt);
        let active = view.sim.active();
        let present = self.present(i);
        let wait = self.outs[i]
            .pacer
            .presented(present, active, Instant::now());
        self.arm(i, wait);
    }

    fn wake(&mut self, i: usize) {
        let wait = self.outs[i].pacer.wake(Instant::now());
        self.arm(i, wait);
    }

    /// Ambient flow is app-wide: it applies to every output and survives
    /// rebuilds (resize, scale changes), which start from `cfg.params`.
    fn toggle_ambient(&mut self) {
        self.cfg.params.ambient_on.toggle();
        for i in 0..self.outs.len() {
            if let Some(view) = self.outs[i].view.as_mut() {
                view.sim.p.ambient_on = self.cfg.params.ambient_on;
            }
            self.wake(i);
        }
    }

    fn pointer_event(&mut self, i: usize, ev: &PointerEvent) {
        if let PointerEventKind::Press {
            button: BTN_MIDDLE, ..
        } = ev.kind
        {
            return self.toggle_ambient();
        }
        let out = &mut self.outs[i];
        let Some(view) = out.view.as_mut() else {
            return;
        };
        let s = out.scale;
        let (cx, cy) = view
            .renderer
            .layout
            .to_cell(ev.position.0 * s, ev.position.1 * s);
        match ev.kind {
            PointerEventKind::Enter { serial } => {
                out.cursor = Some((cx, cy));
                if let Some(dev) = &self.cursor_device {
                    dev.set_shape(serial, Shape::Default);
                }
                return;
            }
            PointerEventKind::Leave { .. } => {
                out.cursor = None;
                return;
            }
            PointerEventKind::Motion { .. } => {
                if let Some((px, py)) = out.cursor {
                    view.sim.push(px, py, cx, cy);
                }
                out.cursor = Some((cx, cy));
            }
            PointerEventKind::Press { button, .. } => match button {
                BTN_LEFT => view.sim.burst(cx, cy, 40.0),
                BTN_RIGHT => view.sim.burst(cx, cy, -40.0),
                _ => return,
            },
            PointerEventKind::Axis { vertical, .. } => {
                let d = if vertical.discrete != 0 {
                    vertical.discrete as f32
                } else {
                    vertical.absolute as f32 / 10.0
                };
                if d == 0.0 {
                    return;
                }
                view.sim.swirl(cx, cy, -d * 25.0);
            }
            PointerEventKind::Release { .. } => return,
        }
        self.wake(i);
    }
}

impl CompositorHandler for App {
    fn scale_factor_changed(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        s: &wl_surface::WlSurface,
        f: i32,
    ) {
        // Integer scale only matters without fractional scaling.
        if let Some(i) = self.find(s)
            && self.outs[i].frac.is_none()
        {
            self.set_scale(i, f64::from(f));
        }
    }

    fn transform_changed(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &wl_surface::WlSurface,
        _: wl_output::Transform,
    ) {
    }

    fn frame(&mut self, _: &Connection, _: &QueueHandle<Self>, s: &wl_surface::WlSurface, _: u32) {
        let Some(i) = self.find(s) else { return };
        let out = &mut self.outs[i];
        let active = out.view.as_ref().is_some_and(|v| v.sim.active());
        let wait = out.pacer.on_frame(active, Instant::now());
        self.arm(i, wait);
    }

    fn surface_enter(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &wl_surface::WlSurface,
        _: &wl_output::WlOutput,
    ) {
    }
    fn surface_leave(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &wl_surface::WlSurface,
        _: &wl_output::WlOutput,
    ) {
    }
}

impl OutputHandler for App {
    fn output_state(&mut self) -> &mut OutputState {
        &mut self.output_state
    }

    fn new_output(&mut self, _: &Connection, qh: &QueueHandle<Self>, output: wl_output::WlOutput) {
        let surface = self.compositor.create_surface(qh);
        let layer = self.layer_shell.create_layer_surface(
            qh,
            surface,
            Layer::Background,
            Some("situ"),
            Some(&output),
        );
        layer.set_anchor(Anchor::all());
        layer.set_exclusive_zone(-1);
        layer.set_keyboard_interactivity(KeyboardInteractivity::None);
        layer.set_size(0, 0);
        // Fully opaque: compositors can skip blending/drawing what's below.
        if let Ok(opaque) = Region::new(&self.compositor) {
            opaque.add(0, 0, i32::MAX, i32::MAX);
            layer
                .wl_surface()
                .set_opaque_region(Some(opaque.wl_region()));
        }
        // Static: take no input at all (empty input region).
        if !self.cfg.interactive
            && let Ok(none) = Region::new(&self.compositor)
        {
            layer.wl_surface().set_input_region(Some(none.wl_region()));
        }
        let frac = self.fractional.as_ref().map(|(fm, vp)| {
            let s = layer.wl_surface();
            (
                fm.get_fractional_scale(s, qh, s.id()),
                vp.get_viewport(s, qh, ()),
            )
        });
        layer.commit();
        let scale = f64::from(
            self.output_state
                .info(&output)
                .map_or(1, |i| i.scale_factor),
        );
        self.outs.push(Out {
            output,
            layer,
            scale,
            frac,
            size: (0, 0),
            view: None,
            pacer: Pacer::new(self.interval(), Instant::now()),
            timer: None,
            cursor: None,
        });
    }

    fn update_output(&mut self, _: &Connection, _: &QueueHandle<Self>, _: wl_output::WlOutput) {}

    fn output_destroyed(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        output: wl_output::WlOutput,
    ) {
        if let Some(i) = self.outs.iter().position(|o| o.output == output) {
            self.remove(i);
        }
    }
}

impl LayerShellHandler for App {
    fn closed(&mut self, _: &Connection, _: &QueueHandle<Self>, layer: &LayerSurface) {
        if let Some(i) = self.outs.iter().position(|o| &o.layer == layer) {
            self.remove(i);
        }
    }

    fn configure(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        layer: &LayerSurface,
        c: LayerSurfaceConfigure,
        _: u32,
    ) {
        let Some(i) = self.outs.iter().position(|o| &o.layer == layer) else {
            return;
        };
        // Static outputs drop their view after drawing; only redraw on resize.
        let first = self.outs[i].view.is_none() && self.cfg.interactive;
        if self.outs[i].size != c.new_size || first {
            self.outs[i].size = c.new_size;
            self.rebuild(i);
        }
    }
}

impl SeatHandler for App {
    fn seat_state(&mut self) -> &mut SeatState {
        &mut self.seat_state
    }

    fn new_seat(&mut self, _: &Connection, _: &QueueHandle<Self>, _: wl_seat::WlSeat) {}

    fn new_capability(
        &mut self,
        _: &Connection,
        qh: &QueueHandle<Self>,
        seat: wl_seat::WlSeat,
        cap: Capability,
    ) {
        if cap == Capability::Pointer && self.pointer.is_none() && self.cfg.interactive {
            let Ok(pointer) = self.seat_state.get_pointer(qh, &seat) else {
                return;
            };
            self.cursor_device = self
                .cursor_shape
                .as_ref()
                .map(|m| m.get_shape_device(&pointer, qh));
            self.pointer = Some(pointer);
        }
    }

    fn remove_capability(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: wl_seat::WlSeat,
        cap: Capability,
    ) {
        if cap == Capability::Pointer {
            if let Some(d) = self.cursor_device.take() {
                d.destroy();
            }
            if let Some(p) = self.pointer.take() {
                p.release();
            }
        }
    }

    fn remove_seat(&mut self, _: &Connection, _: &QueueHandle<Self>, _: wl_seat::WlSeat) {}
}

impl PointerHandler for App {
    fn pointer_frame(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &wl_pointer::WlPointer,
        events: &[PointerEvent],
    ) {
        for ev in events {
            if let Some(i) = self.find(&ev.surface) {
                self.pointer_event(i, ev);
            }
        }
    }
}

impl Dispatch<WpFractionalScaleV1, ObjectId> for App {
    fn event(
        app: &mut Self,
        _: &WpFractionalScaleV1,
        event: wp_fractional_scale_v1::Event,
        surface: &ObjectId,
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        if let wp_fractional_scale_v1::Event::PreferredScale { scale } = event
            && let Some(i) = app
                .outs
                .iter()
                .position(|o| &o.layer.wl_surface().id() == surface)
        {
            app.set_scale(i, f64::from(scale) / 120.0);
        }
    }
}

/// Event-less globals/objects.
macro_rules! ignore_events {
    ($($ty:ty, $data:ty);*) => {$(
        impl Dispatch<$ty, $data> for App {
            fn event(
                _: &mut Self,
                _: &$ty,
                _: <$ty as Proxy>::Event,
                _: &$data,
                _: &Connection,
                _: &QueueHandle<Self>,
            ) {
            }
        }
    )*};
}
ignore_events!(WpFractionalScaleManagerV1, (); WpViewporter, (); WpViewport, ());

impl ShmHandler for App {
    fn shm_state(&mut self) -> &mut Shm {
        &mut self.shm
    }
}

delegate_compositor!(App);
delegate_output!(App);
delegate_shm!(App);
delegate_seat!(App);
delegate_pointer!(App);
delegate_layer!(App);
delegate_registry!(App);

impl ProvidesRegistryState for App {
    fn registry(&mut self) -> &mut RegistryState {
        &mut self.registry_state
    }
    registry_handlers![OutputState, SeatState];
}
