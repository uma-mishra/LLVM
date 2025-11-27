
### Note ::
```
The PTX or CUBIN file runs on GPU and not on CPU.
When we load PTX via CUDA driver API (cuModuleLoadDataEx()) the GPU driver JIT compiler compiles it into SASS (native GPU machine code) for the current GPU Architecture.
CUBIN is a precompiled binary containing SASS. It avoids JIT compilation.

```


```
While installing NVIDA-toolkit it used to take the same path as nvidia-smi.
which nvidia-smi
/usr/bin
nvcc --version
......
It get replaced by the installation of the nvidia toolkit and can show blank.
So we can put the NVCC in a different path and keep nvidia-smi in /usr/bin



```


```
ashwina@ashwina-Precision-T1650:~/llvm-experiments/vector_add$ sudo sh cuda_12.9.0_575.51.03_linux.run
[sudo] password for ashwina: 
	===========
= Summary =
===========

Driver:   Not Selected
Toolkit:  Installed in /usr/local/cuda-12.9/

Please make sure that
 -   PATH includes /usr/local/cuda-12.9/bin
 -   LD_LIBRARY_PATH includes /usr/local/cuda-12.9/lib64, or, add /usr/local/cuda-12.9/lib64 to /etc/ld.so.conf and run ldconfig as root

To uninstall the CUDA Toolkit, run cuda-uninstaller in /usr/local/cuda-12.9/bin
***WARNING: Incomplete installation! This installation did not install the CUDA Driver. A driver of version at least 575.00 is required for CUDA 12.9 functionality to work.
To install the driver using this installer, run the following command, replacing <CudaInstaller> with the name of this run file:
    sudo <CudaInstaller>.run --silent --driver

Logfile is /var/log/cuda-installer.log



```


Steps to run CUDA PTX on NVIDA GPU

```

1.  /usr/local/cuda-10.2/bin/nvcc -ptx demo.cu -o demo.ptx
2.  /usr/local/cuda-10.2/bin/nvcc demo_run.cu -o run_ptx -lcuda 
3.  ./run_ptx


```

