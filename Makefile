# mojo-bench: `make bench` builds everything and reproduces results/.

PYTHON ?= python3
ifeq ($(origin CC),default)
CC     := gcc
endif
# GCC's default dialect (gnu17) is used on purpose: -std=c11 would switch off
# floating-point contraction, i.e. FMA, which -march=native otherwise allows.
CFLAGS ?= -O3 -march=native -Wall -Wextra -Wno-unknown-pragmas
MOJO   ?= mojo
BUILD  := build

.PHONY: all build bench quick plot scaling roofline clean deps

all: build

deps:
	$(PYTHON) -m pip install -r requirements.txt

build: $(BUILD)/kernels_c $(BUILD)/kernels_c_omp $(BUILD)/kernels_mojo $(BUILD)/roofline_probe

$(BUILD)/kernels_c: kernels/c/kernels.c
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -o $@ $< -lm

$(BUILD)/kernels_c_omp: kernels/c/kernels.c
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -fopenmp -o $@ $< -lm

$(BUILD)/roofline_probe: kernels/c/roofline_probe.c
	@mkdir -p $(@D)
	$(CC) -O3 -march=native -mprefer-vector-width=512 -fopenmp -Wall -Wextra -o $@ $<

$(BUILD)/kernels_mojo: kernels/mojo/kernels.mojo
	@mkdir -p $(@D)
	$(MOJO) build -O3 $< -o $@

# Everything in results/: main comparison, thread scaling and roofline.
bench: build
	$(PYTHON) bench/run.py --profile full --out results
	$(PYTHON) bench/plot.py results
	$(PYTHON) bench/scaling.py --out results
	$(PYTHON) bench/roofline.py results

# Individual parts.
scaling: build
	$(PYTHON) bench/scaling.py --out results

roofline: build
	$(PYTHON) bench/roofline.py results

# Small sizes, single repeat: a correctness check, not a benchmark (used in CI).
quick: build
	$(PYTHON) bench/run.py --profile quick --out $(BUILD)/quick --no-startup

plot:
	$(PYTHON) bench/plot.py results

clean:
	rm -rf $(BUILD)
