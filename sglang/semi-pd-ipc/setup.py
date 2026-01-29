from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="semi-pd-ipc",
    version="0.1.0",
    ext_modules=[
        CUDAExtension(
            name="semi_pd_ipc",
            sources=["ipc.cpp", "allocate_vexpert_cuda.cu", "replicate_vexpert_cuda.cu", "allocate_vexpert_optimization_cuda.cu"],
            extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3"]},
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)