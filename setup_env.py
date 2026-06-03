"""
setup_env.py
------------
Skrip verifikasi environment untuk proyek Essay Scoring Multimodal.
Jalankan: python setup_env.py
"""

import subprocess, sys, importlib, platform, shutil


REQUIRED = {
    # Core ML
    "torch"          : "torch",
    "torchvision"    : "torchvision",
    "transformers"   : "transformers",
    "sklearn"        : "scikit-learn",
    "numpy"          : "numpy",
    "pandas"         : "pandas",
    "PIL"            : "Pillow",
    # OCR
    "easyocr"        : "easyocr",
    # Visualisasi & monitoring
    "matplotlib"     : "matplotlib",
    "seaborn"        : "seaborn",
    "tqdm"           : "tqdm",
    # Experiment tracking
    "wandb"          : "wandb",
    # App
    "streamlit"      : "streamlit",
}

def check(pkg_import, pkg_install):
    try:
        mod = importlib.import_module(pkg_import)
        ver = getattr(mod, "__version__", "?")
        print(f"  [OK] {pkg_install:<25} v{ver}")
        return True
    except ImportError:
        print(f"  [--] {pkg_install:<25} — tidak terinstall")
        return False


def main():
    print("=" * 55)
    print(f"  Python  : {sys.version.split()[0]}")
    print(f"  OS      : {platform.system()} {platform.release()}")
    print("=" * 55)

    missing = []
    for imp, pkg in REQUIRED.items():
        if not check(imp, pkg):
            missing.append(pkg)

    # Cek CUDA
    print()
    try:
        import torch
        cuda = torch.cuda.is_available()
        print(f"  CUDA    : {'tersedia — ' + torch.cuda.get_device_name(0) if cuda else 'tidak tersedia (CPU mode)'}")
    except Exception:
        pass

    if missing:
        print(f"\n  {len(missing)} library belum terinstall.")
        ans = input("  Install sekarang? [y/n]: ").strip().lower()
        if ans == "y":
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", *missing,
                "--quiet", "--upgrade"
            ])
            print("  Instalasi selesai. Jalankan ulang untuk verifikasi.")
    else:
        print("\n  Semua library siap!")

if __name__ == "__main__":
    main()
