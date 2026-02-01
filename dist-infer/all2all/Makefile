.PHONY: all config build

TARGET ?= gfx942

BUILD_TYPE ?= RelWithDebInfo
BUILD_DIR ?= build

PYTHON_DIR ?= $(shell python -c "import site; print(site.getsitepackages()[0])")

all: config build submit

config:
	PYTORCH_ROCM_ARCH=$(TARGET) cmake -B $(BUILD_DIR) . \
		-DCMAKE_BUILD_TYPE=$(BUILD_TYPE) \
		-DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
		-DCMAKE_PREFIX_PATH="/opt/rocm;$(PYTHON_DIR)" \
		-DCMAKE_HIP_ARCHITECTURES=$(TARGET) \
		-DGPU_TARGETS=$(TARGET) \
		-DAMDGPU_TARGETS=$(TARGET) \
		-G Ninja

build:
	cmake --build $(BUILD_DIR) -j8

test: build
	PYTHONPATH=$(PYTHONPATH):$(realpath tools):$(realpath $(BUILD_DIR)) python tools/smoke_test.py

clean:
	rm -r $(BUILD_DIR)

local:
	python submit.py local_test
	POPCORN_GPUS=2 POPCORN_FD=2 python eval.py benchmark benchmark.txt

submit:
	python submit.py

dis:
	roc-obj -d build/all2all.so
