# -*- mode: python ; coding: utf-8 -*-
# Build with:  pyinstaller vision_app.spec
# Output executable will appear in dist/SignSync/SignSync.exe

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[('app_icon.ico', '.')],
    hiddenimports=['cv2', 'mediapipe'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='SignSync',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,   # set True temporarily if you need to see errors
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='app_icon.ico',
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='SignSync',
)
