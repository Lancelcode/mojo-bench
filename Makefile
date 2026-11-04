# mojo-bench: `make bench` builds everything and reproduces results/.

PYTHON ?= python3
ifeq ($(origin CC),default)
CC     := gcc
endif
CFLAGS ?= -O3 -march=native -std=c11 -Wall -Wextra -Wno-unknown-pragmas
MOJO   ?= mojo
BUILD  := build

.PHONY: all build bench quick plot clean deps

all: build

deps:
	$(PYTHON) -m pip install -r requirements.txt

build: $(BUILD)/kernels_c $(BUILD)/kernels_mojo

$(BUILD)/kernels_c: kernels/c/kernels.c
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -o $@ $< -lm

$(BUILD)/kernels_mojo: kernels/mojo/kernels.mojo
	@mkdir -p $(@D)
	$(MOJO) build -O3 $< -o $@

# Full benchmark: results/results.{csv,md}, machine.json, startup.csv and chart.
bench: build
	$(PYTHON) bench/run.py --profile full --out results
	$(PYTHON) bench/plot.py results

# Small sizes, single repeat: a correctness check, not a benchmark (used in CI).
quick: build
	$(PYTHON) bench/run.py --profile quick --out $(BUILD)/quick --no-startup

plot:
	$(PYTHON) bench/plot.py results

clean:
	rm -rf $(BUILD)
