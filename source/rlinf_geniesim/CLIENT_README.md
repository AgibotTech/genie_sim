# RLinf GenieSim client

`rlinf-geniesim-client` connects RLinf to a separate GenieSim server through
ZMQ. It provides the existing `rlinf_geniesim.comm` client API and
`rlinf_geniesim.envs.vec_env.GenieSimVecEnv` import path. SHM is available for
clients on the same host as the server.

The package contains the transport clients, vector environment, and sensor
randomization helpers. On Linux, the optional
[`spawn_server` API](docs/interface.md#private-local-servers) manages a private
server using a separately prepared local checkout and runtime. It does not
include server implementations, launch
scripts, simulation backends, benchmark tasks, or assets. The server runs from
the GenieSim repository and supplies task metadata and observations.

## Install from source

Use Python 3.10 or newer. From the GenieSim repository root, install into the
Python environment that runs RLinf:

```bash
python -m pip install ./source/rlinf_geniesim
```

This is a regular installation. After installation, the client does not need
the source checkout or a `PYTHONPATH` entry. A client image can install the
package during its build and omit the checkout from the final image. Keep the
source revision used to build that image aligned with the server deployment.

The base dependencies are NumPy, PyZMQ, and Pillow. RLinf supplies PyTorch and
its model-specific dependency constraints; this package does not install or
select a PyTorch version. Without PyTorch, the vector environment can return
NumPy observations.

Install optional dependencies as needed:

```bash
# Local SHM communication; requires a host with POSIX shared memory and semaphores.
python -m pip install './source/rlinf_geniesim[shm]'

# Sensor augmentations that use SciPy.
python -m pip install './source/rlinf_geniesim[sensor]'

# Both optional features.
python -m pip install './source/rlinf_geniesim[shm,sensor]'
```

SHM uses an existing task created by the server application. Remote task
initialization, reload, and metadata queries use ZMQ.

## Robot configuration

The vector environment includes the default G2 joint groups. Other robots
must provide their joint groups explicitly when the benchmark package is not
installed. Automatic layout matching against benchmark robot configurations
is available only when `geniesim_benchmark` is separately installed.

## License

The package source is licensed under MPL-2.0; see the included `LICENSE`.
Dependencies and server-side software and assets retain their own terms.
