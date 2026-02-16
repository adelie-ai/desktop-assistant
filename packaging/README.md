# Packaging

This directory contains distro/app packaging definitions for Desktop Assistant.

## Targets

- `debian/` — Debian/Ubuntu `.deb` packaging files
- `arch/` — Arch Linux `PKGBUILD`
- `fedora/` — Fedora/RHEL `.spec`
- `snap/` — Snapcraft manifest
- `flatpak/` — Flatpak manifest + desktop metadata

## Debian

From a source tarball root containing this project:

```bash
cp -r packaging/debian/debian .
dpkg-buildpackage -us -uc -b
```

## Arch

From `packaging/arch` with a matching source tarball in the same directory:

```bash
makepkg -si
```

## Fedora

From `packaging/fedora` with source tarball available as `desktop-assistant-<version>.tar.gz`:

```bash
rpmbuild -ba desktop-assistant.spec
```

## Snap

From repository root:

```bash
snapcraft --destructive-mode --dir packaging/snap
```

## Flatpak

From repository root:

```bash
flatpak-builder --force-clean build-dir packaging/flatpak/org.desktopassistant.App.yml
```

## Notes

- All package definitions currently install:
  - `desktop-assistant-daemon`
  - `desktop-assistant-tui`
  - `systemd/desktop-assistant-daemon.service`
  - `systemd/org.desktopAssistant.service`
- The maintainer/contact fields use placeholders and should be replaced before publication.

## CI and Local Validation

Run the unified packaging checks locally:

```bash
just packaging-check
```

Run release-binary build + packaging checks (same flow as CI):

```bash
just packaging-ci
```

GitHub Actions workflow:

- `.github/workflows/packaging-checks.yml`
- Script entrypoint: `packaging/ci/check-packaging.sh`
