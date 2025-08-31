# -*- mode: python ; coding: utf-8 -*-
import os, sys
from PyInstaller.utils.hooks import collect_dynamic_libs, collect_all

datas = []
binaries = []
hiddenimports = []

binaries += collect_dynamic_libs('cv2')
for pkg in ('numpy','mss'):
    d, b, h = collect_all(pkg)
    datas += d; binaries += b; hiddenimports += h

a = Analysis(
    ['LensDesktop.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

# Use an absolute path to avoid path confusion
icon_path = os.path.abspath('icon.ico')

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='LensDesktop',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=icon_path,          # <-- WINDOWS ICON GOES HERE
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='LensDesktop',
)

# Optional: only used on macOS; safe to keep but ignored on Windows
if sys.platform == 'darwin':
    app = BUNDLE(
        coll,
        name='LensDesktop.app',
        icon='icon.icns',     # .icns for macOS
        bundle_identifier='com.yourdomain.lensdesktop',
        info_plist={
            "NSCameraUsageDescription":
                "LensDesktop needs camera access for OpenCV VideoCapture."
        },
    )