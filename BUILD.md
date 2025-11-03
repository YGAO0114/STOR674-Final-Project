
#### Option 1: Remote Build (Recommended - No root required)

Build on Sylabs Cloud (free account required):

```bash
# First, create account and get token at https://cloud.sylabs.io
singularity remote login
# Build remotely
singularity build --remote stor674.sif Singularity.def
```


```bash
# Run interactively with bash
singularity shell stor674.sif

# Run a Python script
singularity exec stor674.sif python xxxscript.py

# Run with GPU support (if available)
singularity exec --nv stor674.sif python -c "import torch; print(torch.cuda.is_available())"

# Run and mount current directory
singularity exec --bind $(pwd):/workspace stor674.sif bash

# Run with multiple bind mounts
singularity exec --bind $(pwd):/workspace --bind $(pwd)/Data:/workspace/Data stor674.sif bash
```

### GPU Support
```bash
# Check if GPU is available
singularity exec --nv stor674.sif python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

### Common Commands

```bash
# Enter interactive shell
singularity shell stor674.sif

# Run Jupyter (mount current directory)
singularity exec --bind $(pwd):/workspace stor674.sif jupyter lab --ip=0.0.0.0 --port=8888 --no-browser

# Run a specific script
singularity exec --bind $(pwd):/workspace stor674.sif python scripts/01-segmentation/preprocess/preprocess.py

# Check installed packages
singularity exec stor674.sif pip list
```

### Building from requirements.txt (Alternative)
If you prefer to use the requirements.txt file during build, you can modify the Singularity.def file to copy and use it:

```bash
# In the %post section, add:
COPY requirements.txt /tmp/requirements.txt
pip3 install --no-cache-dir -r /tmp/requirements.txt
```

## Troubleshooting

### Singularity Build Issues

**Issue: "FATAL: --remote, --fakeroot, or the proot command are required to build this source as a non-root user"**

1. **Remote build (easiest):**
   ```bash
   singularity remote login  # First time only
   singularity build --remote stor674.sif Singularity.def
   ```

2. **Fakeroot (if configured):**
   ```bash
   singularity build --fakeroot stor674.sif Singularity.def
   ```



### Verification

After building, verify the container:

```bash
# Check Python version
singularity exec stor674.sif python --version

# Check PyTorch and CUDA
singularity exec --nv stor674.sif python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Next Steps

1. Build the container: `singularity build stor674.sif Singularity.def`
2. Test GPU access: `singularity exec --nv stor674.sif python -c "import torch; print(torch.cuda.is_available())"`
3. Run your scripts: `singularity exec --bind $(pwd):/workspace stor674.sif python your_script.py`

## Notes
- Use `--bind` to mount directories from the host
- Use `--nv` flag for GPU/NVIDIA support
- The .sif file is portable and can be copied to other systems

