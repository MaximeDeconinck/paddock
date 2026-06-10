//! macOS menu bar item showing live serve endpoints. The menu content is
//! computed in `menu_model` (pure, tested); this module owns the thin
//! tray-icon/tao/arboard rendering layer (manually smoke-tested).

pub mod menu_model;

#[cfg(target_os = "macos")]
mod macos {
    use std::collections::HashMap;
    use std::time::{Duration, Instant};

    use tao::event::{Event, StartCause};
    use tao::event_loop::{ControlFlow, EventLoopBuilder};
    use tetro_core::hardware::runtimes::detect_runtimes;
    use tetro_core::hardware::RealSystemProbe;
    use tetro_core::serving::{ollama_loaded_models, Registry};
    use tray_icon::menu::{Menu, MenuEvent, MenuId, MenuItem, PredefinedMenuItem};
    use tray_icon::{TrayIcon, TrayIconBuilder};

    use super::menu_model::{build_menu_model, MenuEntry, MenuModel};

    const REFRESH_EVERY: Duration = Duration::from_secs(5);

    enum UserEvent {
        Menu(MenuEvent),
    }

    /// What activating a menu item does; keyed by tray-icon menu id because
    /// the menu is rebuilt from scratch on every refresh.
    enum Action {
        Copy(String),
        Refresh,
        Quit,
    }

    pub fn run() -> anyhow::Result<()> {
        let event_loop = EventLoopBuilder::<UserEvent>::with_user_event().build();
        // Forward menu events through the proxy so the tao loop wakes up
        // (tray-icon README pattern for tao/winit).
        let proxy = event_loop.create_proxy();
        MenuEvent::set_event_handler(Some(move |e| {
            let _ = proxy.send_event(UserEvent::Menu(e));
        }));

        let mut tray: Option<TrayIcon> = None;
        let mut actions: HashMap<MenuId, Action> = HashMap::new();
        let mut next_refresh = Instant::now() + REFRESH_EVERY;

        event_loop.run(move |event, _, control_flow| {
            match event {
                // Create the tray icon once the loop actually runs
                // (tray-icon issue #90: creating it earlier can hide it).
                Event::NewEvents(StartCause::Init) => {
                    let (menu, map) = render(&refresh_model());
                    actions = map;
                    tray = Some(
                        TrayIconBuilder::new()
                            .with_menu(Box::new(menu))
                            .with_icon(template_icon())
                            .with_icon_as_template(true)
                            .with_tooltip("tetro — serve endpoints")
                            .build()
                            .expect("failed to create the menu bar item"),
                    );
                    next_refresh = Instant::now() + REFRESH_EVERY;
                }
                Event::NewEvents(_) if Instant::now() >= next_refresh => {
                    refresh_into(tray.as_ref(), &mut actions);
                    next_refresh = Instant::now() + REFRESH_EVERY;
                }
                Event::UserEvent(UserEvent::Menu(e)) => match actions.get(e.id()) {
                    Some(Action::Copy(url)) => copy_to_clipboard(url),
                    Some(Action::Refresh) => {
                        refresh_into(tray.as_ref(), &mut actions);
                        next_refresh = Instant::now() + REFRESH_EVERY;
                    }
                    Some(Action::Quit) => {
                        tray = None; // remove the icon before exiting
                        *control_flow = ControlFlow::Exit;
                        return;
                    }
                    None => {}
                },
                _ => {}
            }
            *control_flow = ControlFlow::WaitUntil(next_refresh);
        })
    }

    /// Probe the world: installed runtimes, live registry records, Ollama ps.
    fn refresh_model() -> MenuModel {
        let probe = RealSystemProbe;
        let runtimes = detect_runtimes(&probe);
        let records = Registry::open_default().list_live(&probe);
        let ps = ollama_loaded_models(&probe);
        build_menu_model(&records, ps.as_deref(), &runtimes)
    }

    fn refresh_into(tray: Option<&TrayIcon>, actions: &mut HashMap<MenuId, Action>) {
        let (menu, map) = render(&refresh_model());
        *actions = map;
        if let Some(t) = tray {
            t.set_menu(Some(Box::new(menu)));
        }
    }

    /// MenuModel → tray-icon menu + id→action map. Header/Info are disabled
    /// items; Model rows are clickable and copy their URL. The footer
    /// (separator + Rafraîchir + Quitter) is owned by this layer.
    fn render(model: &MenuModel) -> (Menu, HashMap<MenuId, Action>) {
        let menu = Menu::new();
        let mut actions = HashMap::new();
        let append = |item: &dyn tray_icon::menu::IsMenuItem| {
            menu.append(item).expect("appending tray menu item");
        };
        for entry in &model.entries {
            match entry {
                MenuEntry::Header(text) | MenuEntry::Info(text) => {
                    append(&MenuItem::new(text, false, None));
                }
                MenuEntry::Model { label, copy_url } => {
                    let item = MenuItem::new(label, true, None);
                    actions.insert(item.id().clone(), Action::Copy(copy_url.clone()));
                    append(&item);
                }
                MenuEntry::Separator => append(&PredefinedMenuItem::separator()),
            }
        }
        append(&PredefinedMenuItem::separator());
        let refresh = MenuItem::new("Rafraîchir", true, None);
        actions.insert(refresh.id().clone(), Action::Refresh);
        append(&refresh);
        let quit = MenuItem::new("Quitter tetro tray", true, None);
        actions.insert(quit.id().clone(), Action::Quit);
        append(&quit);
        (menu, actions)
    }

    fn copy_to_clipboard(text: &str) {
        let res = arboard::Clipboard::new().and_then(|mut c| c.set_text(text.to_string()));
        if let Err(e) = res {
            eprintln!("could not copy to clipboard: {e}");
        }
    }

    /// 22×22 monochrome "t" glyph, black + alpha. Rendered as a template
    /// image so macOS adapts it to light/dark menu bars.
    fn template_icon() -> tray_icon::Icon {
        const SIZE: usize = 22;
        #[rustfmt::skip]
        const GLYPH: [&str; SIZE] = [
            "                      ",
            "                      ",
            "                      ",
            "        ###           ",
            "        ###           ",
            "        ###           ",
            "   ##############     ",
            "   ##############     ",
            "   ##############     ",
            "        ###           ",
            "        ###           ",
            "        ###           ",
            "        ###           ",
            "        ###           ",
            "        ###           ",
            "        ###      #    ",
            "        ####    ##    ",
            "         ##########   ",
            "          ########    ",
            "            ####      ",
            "                      ",
            "                      ",
        ];
        let mut rgba = Vec::with_capacity(SIZE * SIZE * 4);
        for row in GLYPH {
            for c in row.chars() {
                let a = if c == '#' { 255 } else { 0 };
                rgba.extend_from_slice(&[0, 0, 0, a]);
            }
        }
        tray_icon::Icon::from_rgba(rgba, SIZE as u32, SIZE as u32)
            .expect("static icon buffer is well-formed")
    }
}

#[cfg(target_os = "macos")]
pub fn run() -> anyhow::Result<()> {
    macos::run()
}

#[cfg(not(target_os = "macos"))]
pub fn run() -> anyhow::Result<()> {
    anyhow::bail!("tetro tray is macOS-only for now")
}
