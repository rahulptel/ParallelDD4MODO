# TSP GPU Status

Date: 2026-06-16

Binary:

```bash
bin/multiobj_nobjs5_gpu
```

Command pattern:

```bash
./bin/multiobj_nobjs5_gpu <instance> 3 1 0 --backend gpu --kernel 3
```

This uses TSP (`problem_type=3`), top-down enumeration (`method=1`), no state
dominance, and the GPU backend.

The sandboxed runner could not see the CUDA device. Running the same binary with
host GPU access succeeded, except where the CUDA allocator reported out of
memory.

## Results

| Instance | Status | Solutions | CPU total (s) | CPU compile (s) | CPU enumeration (s) | Wall compile (s) | Wall enumeration (s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `data/5/tsp/tsp-nobj5-ncities15-seed1081.dat` | Completed | 55198 | 185.683 | 0.400584 | 185.282 | 0.401231 | 185.525 |
| `data/5/tsp/tsp-nobj5-ncities15-seed11093.dat` | Completed | 29505 | 41.0364 | 0.443722 | 40.5926 | 0.451828 | 40.6533 |
| `data/5/tsp/tsp-nobj5-ncities15-seed13095.dat` | Memory out | - | - | - | - | - | - |
| `data/5/tsp/tsp-nobj5-ncities15-seed14792.dat` | Completed | 77303 | 221.912 | 0.449051 | 221.463 | 0.456132 | 221.731 |
| `data/5/tsp/tsp-nobj5-ncities15-seed23138.dat` | Completed | 51306 | 90.9773 | 0.342015 | 90.6353 | 0.347557 | 90.7084 |
| `data/5/tsp/tsp-nobj5-ncities15-seed24645.dat` | Completed | 38883 | 102.92 | 0.42265 | 102.497 | 0.430411 | 102.657 |
| `data/5/tsp/tsp-nobj5-ncities15-seed30131.dat` | Completed | 24381 | 60.2311 | 0.391112 | 59.84 | 0.395054 | 59.9564 |
| `data/5/tsp/tsp-nobj5-ncities15-seed32239.dat` | Completed | 78849 | 254.611 | 0.328798 | 254.282 | 0.328811 | 254.694 |
| `data/5/tsp/tsp-nobj5-ncities15-seed6440.dat` | Completed | 32237 | 65.1173 | 0.556459 | 64.5608 | 0.559308 | 64.7564 |
| `data/5/tsp/tsp-nobj5-ncities15-seed7702.dat` | Completed | 53917 | 187.309 | 0.41342 | 186.896 | 0.414335 | 187.051 |

## CPU Coupled Follow-up

The instance that memory-outed on GPU top-down was rerun with CPU coupled
enumeration using 16 threads:

```bash
./bin/multiobj_nobjs5_cpu ./data/5/tsp/tsp-nobj5-ncities15-seed13095.dat 3 3 0 --backend cpu --cpu-threads 16 --cpu-kernel 3
```

| Instance | Backend | Method | Threads | Status | Solutions | CPU total (s) | CPU compile (s) | CPU enumeration (s) | Wall compile (s) | Wall enumeration (s) |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `data/5/tsp/tsp-nobj5-ncities15-seed13095.dat` | CPU | Coupled (`method=3`) | 16 | Completed | 117171 | 2548.99 | 0.496982 | 2548.5 | 0.330581 | 393.197 |

## Memory-out Buckets

Memory out:

- `data/5/tsp/tsp-nobj5-ncities15-seed13095.dat`
  - Error:
    ```text
    thrust::system::detail::bad_alloc
    std::bad_alloc: cudaErrorMemoryAllocation: out of memory
    ```

Did not memory out:

- `data/5/tsp/tsp-nobj5-ncities15-seed1081.dat`
- `data/5/tsp/tsp-nobj5-ncities15-seed11093.dat`
- `data/5/tsp/tsp-nobj5-ncities15-seed14792.dat`
- `data/5/tsp/tsp-nobj5-ncities15-seed23138.dat`
- `data/5/tsp/tsp-nobj5-ncities15-seed24645.dat`
- `data/5/tsp/tsp-nobj5-ncities15-seed30131.dat`
- `data/5/tsp/tsp-nobj5-ncities15-seed32239.dat`
- `data/5/tsp/tsp-nobj5-ncities15-seed6440.dat`
- `data/5/tsp/tsp-nobj5-ncities15-seed7702.dat`

## Run Artifacts

The five remaining runs from 2026-06-16 wrote stdout, stderr, and
`/usr/bin/time` output under:

```bash
reports/tsp5_15_gpu_remaining_20260616/
```
