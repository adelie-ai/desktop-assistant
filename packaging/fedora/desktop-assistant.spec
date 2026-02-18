Name:           desktop-assistant
Version:        0.1.0
Release:        1%{?dist}
Summary:        Desktop assistant daemon and terminal UI
License:        AGPL-3.0-or-later
URL:            https://example.com/desktop-assistant
Source0:        %{name}-%{version}.tar.gz

BuildRequires:  cargo
BuildRequires:  rust
BuildRequires:  systemd-rpm-macros
Requires:       dbus
Requires:       systemd

%description
Desktop Assistant provides a user-session D-Bus daemon and a terminal UI,
with modular LLM connector backends and MCP integration.

%prep
%autosetup -n %{name}-%{version}

%build
cargo build --release --workspace

%install
install -Dpm0755 target/release/desktop-assistant-daemon %{buildroot}%{_bindir}/desktop-assistant-daemon
install -Dpm0755 target/release/desktop-assistant-tui %{buildroot}%{_bindir}/desktop-assistant-tui
install -Dpm0644 systemd/desktop-assistant-daemon.service %{buildroot}%{_userunitdir}/desktop-assistant-daemon.service
install -Dpm0644 systemd/org.desktopAssistant.service %{buildroot}%{_datadir}/dbus-1/services/org.desktopAssistant.service

%files
%license LICENSE
%doc README.md
%{_bindir}/desktop-assistant-daemon
%{_bindir}/desktop-assistant-tui
%{_userunitdir}/desktop-assistant-daemon.service
%{_datadir}/dbus-1/services/org.desktopAssistant.service

%changelog
* Mon Feb 16 2026 Desktop Assistant Maintainers <packaging@example.com> - 0.1.0-1
- Initial Fedora RPM packaging
