set -eu
area=/build/bf16-store-toolchain
test ! -e "$area"
mkdir -p "$area/packages" "$area/root"
cd "$area/packages"
apt-get download cuda-nvcc-12-9=12.9.86-1 cuda-nvvm-12-9=12.9.86-1 cuda-crt-12-9=12.9.86-1 cuda-cudart-dev-12-9=12.9.79-1 cuda-cudart-12-9=12.9.79-1 cuda-cccl-12-9=12.9.27-1 cuda-driver-dev-12-9=12.9.79-1 libcublas-dev-12-9 libcublas-12-9
for package in ./*.deb; do
    dpkg-deb -f "$package" Package Version
    sha256sum "$package"
    dpkg-deb -x "$package" "$area/root"
done
"$area/root/usr/local/cuda-12.9/bin/nvcc" --version
test -e "$area/root/usr/local/cuda-12.9/include/cuda_runtime.h"
test -e "$area/root/usr/local/cuda-12.9/include/cublas_v2.h"
test -e "$area/root/usr/local/cuda-12.9/lib64/libcublas.so"
