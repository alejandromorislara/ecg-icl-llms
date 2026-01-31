"""
Script de diagnóstico para verificar la instalación de CUDA y PyTorch.

Ejecuta este script para ver el estado de tu configuración:
    python scripts/check_cuda.py
"""

import sys


def check_cuda():
    """Check CUDA availability and configuration."""
    
    print("=" * 60)
    print("Diagnostico de CUDA y PyTorch")
    print("=" * 60)
    
    # Check PyTorch installation
    print("\n1. Verificando instalacion de PyTorch...")
    try:
        import torch
        print(f"   OK PyTorch instalado: {torch.__version__}")
    except ImportError:
        print(f"   ERROR PyTorch NO instalado")
        print(f"\n   Para instalar PyTorch con CUDA:")
        print(f"   pip install torch --index-url https://download.pytorch.org/whl/cu121")
        sys.exit(1)
    
    # Check CUDA availability
    print("\n2. Verificando disponibilidad de CUDA...")
    cuda_available = torch.cuda.is_available()
    
    if cuda_available:
        print(f"   OK CUDA disponible")
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   Devices: {torch.cuda.device_count()}")
        print(f"   Current device: {torch.cuda.current_device()}")
        
        # Memory info
        mem_info = torch.cuda.get_device_properties(0)
        total_memory = mem_info.total_memory / 1e9
        print(f"   VRAM total: {total_memory:.1f} GB")
        
        # Test tensor creation
        print("\n3. Probando creacion de tensor en GPU...")
        try:
            x = torch.randn(100, 100).cuda()
            print(f"   OK Tensor creado correctamente en GPU")
        except Exception as e:
            print(f"   ERROR al crear tensor: {e}")
    else:
        print(f"   ERROR CUDA NO disponible")
        print(f"\n   PyTorch version: {torch.__version__}")
        cuda_version = torch.version.cuda if hasattr(torch.version, 'cuda') else None
        print(f"   CUDA built with: {cuda_version if cuda_version else 'CPU-only build'}")
        
        print(f"\n   Posibles causas:")
        print(f"   1. PyTorch instalado sin soporte CUDA")
        print(f"   2. Drivers NVIDIA no instalados o desactualizados")
        print(f"   3. CUDA toolkit no instalado")
        
        print(f"\n   Soluciones:")
        print(f"\n   A. Verificar drivers NVIDIA:")
        print(f"      nvidia-smi")
        
        print(f"\n   B. Reinstalar PyTorch con CUDA 12.1:")
        print(f"      pip uninstall torch")
        print(f"      pip install torch --index-url https://download.pytorch.org/whl/cu121")
        
        print(f"\n   C. Para CUDA 11.8:")
        print(f"      pip install torch --index-url https://download.pytorch.org/whl/cu118")
        
        print(f"\n   D. Verificar en https://pytorch.org/get-started/locally/")
    
    # Check other important libraries
    print("\n4. Verificando otras librerias...")
    
    libraries = [
        ('transformers', 'transformers'),
        ('peft', 'peft'),
        ('trl', 'trl'),
        ('bitsandbytes', 'bitsandbytes'),
        ('datasets', 'datasets'),
        ('accelerate', 'accelerate'),
    ]
    
    for import_name, display_name in libraries:
        try:
            lib = __import__(import_name)
            version = getattr(lib, '__version__', 'unknown')
            print(f"   OK {display_name}: {version}")
        except ImportError:
            print(f"   ERROR {display_name}: NO instalado")
    
    print("\n" + "=" * 60)
    if cuda_available:
        print("OK Sistema listo para fine-tuning con GPU")
    else:
        print("WARNING Sistema configurado para CPU solamente")
    print("=" * 60)
    print()


if __name__ == "__main__":
    check_cuda()
