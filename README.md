# cuMODD

`cuMODD` is a C++ project for multi-objective optimization using decision diagrams, with CPU and optional CUDA execution paths.

## Build

From the project root:

```bash
make cpu NUM_OBJS=3
```

Optional CUDA build (`nvcc >= 12`):

```bash
make gpu NUM_OBJS=3
```

Build both variants:

```bash
make both NUM_OBJS=3
```

Clean build artifacts:

```bash
make clean
```

## Run

Executable names follow:

- `multiobj_cpu_nobjs<NUM_OBJS>`
- `multiobj_gpu_nobjs<NUM_OBJS>`

CLI:

```bash
./multiobj_cpu_nobjs3 <input-file> <problem-type> <preprocess> <method> <appr-S> <appr-T> <dominance>
```

GPU CLI:

```bash
./multiobj_gpu_nobjs3 <input-file> <problem-type> <preprocess> <method> <appr-S> <appr-T> <dominance>
```

In GPU builds, `method=1` automatically uses CUDA. `method=2` and `method=3` use the CPU implementations.

## Data

Sample/input data is under `data/`.
