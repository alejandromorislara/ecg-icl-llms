# Instalacion de PyTorch con CUDA para Fine-Tuning

## Diagnostico

Antes de instalar, verifica tu configuracion:

```bash
# 1. Verificar GPU y version CUDA
nvidia-smi

# 2. Ejecutar diagnostico del sistema
python scripts/check_cuda.py
```

## Instalacion

### Para CUDA 12.1+ (RTX 3060/3070/3080/3090, RTX 4000 series)

```bash
# 1. Desinstalar PyTorch existente (si lo hay)
pip uninstall torch torchvision torchaudio -y

# 2. Instalar PyTorch con CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. Instalar el resto de dependencias
pip install -r requirements.txt
```

### Para CUDA 11.8

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Verificacion

Despues de instalar, verifica que CUDA este disponible:

```bash
python scripts/check_cuda.py
```

Deberia mostrar:
```
OK CUDA disponible
GPU: NVIDIA GeForce RTX 3060
CUDA version: 12.1
VRAM total: 12.0 GB
```

## Problemas Comunes

### PyTorch no detecta CUDA

**Causa**: PyTorch instalado sin soporte CUDA (version CPU-only)

**Solucion**: Reinstalar con el index-url correcto:
```bash
pip uninstall torch -y
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### bitsandbytes no funciona en Windows

**Causa**: bitsandbytes tiene problemas en Windows

**Solucion**: Usar la version compilada para Windows:
```bash
pip uninstall bitsandbytes -y
pip install bitsandbytes-windows
```

O alternativamente, usar WSL2 con Linux.

### Error "CUDA out of memory"

**Causa**: VRAM insuficiente

**Solucion**: Reducir batch size o gradient accumulation en el config YAML:
```yaml
training:
  per_device_train_batch_size: 1  # Ya es minimo
  gradient_accumulation_steps: 8  # Reducir de 16 a 8
```

## Links Utiles

- [PyTorch Get Started](https://pytorch.org/get-started/locally/)
- [CUDA Toolkit Download](https://developer.nvidia.com/cuda-downloads)
- [NVIDIA Driver Download](https://www.nvidia.com/download/index.aspx)
