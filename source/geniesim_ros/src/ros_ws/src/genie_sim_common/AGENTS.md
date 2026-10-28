# genie_sim_common — AGENTS

Zero-runtime-code package. Every file here is CMake glue consumed
at configure time by other packages in this workspace.

## Layout

```
genie_sim_common/
├── package.xml                          # buildtool-only, no runtime deps
├── CMakeLists.txt                       # installs cmake/ + config-extras
├── genie_sim_common-extras.cmake        # CONFIG_EXTRAS entry point
├── cmake/
│   └── genie_sim_package.cmake          # macros + test wrappers
├── LICENSE
├── README.md
└── AGENTS.md
```

## Wiring

`CMakeLists.txt` calls `ament_package(CONFIG_EXTRAS
"genie_sim_common-extras.cmake")`. That extras file is auto-included
by ament every time a downstream package does
`find_package(genie_sim_common REQUIRED)`, and it in turn `include`s
`${genie_sim_common_DIR}/genie_sim_package.cmake` — which puts
`genie_sim_package()` and the `genie_sim_add_*` test wrappers into
the downstream project's scope.

The `cmake/` directory is installed to
`share/genie_sim_common/cmake/` so the include resolves at install
time; `${genie_sim_common_DIR}` points at
`share/genie_sim_common/cmake` in the install tree.

## Adding to the macro

Edit `cmake/genie_sim_package.cmake`. Rules:

- Prefer `option(...)` and `cmake_parse_arguments(_GS ...)` for opt-in
  behaviour rather than always-on side effects — the macro fires from
  every package, so anything unconditional lands in every build.
- Keep C++-only settings inside an `if(CMAKE_CXX_COMPILER_ID MATCHES
  "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")` guard so
  `project(... NONE)` packages (bringup, engine, planning,
  robot_model, moveit) don't error.
- If you add a new test wrapper (`genie_sim_add_<foo>`), mirror the
  existing pattern: fall back to the plain `ament_add_*` variant when
  the `ament_add_ros_isolated_*` counterpart isn't available.

## Not in scope

- **Python module.** `nav2_common` ships one (rewritten YAML,
  launch-arg helpers); we haven't accumulated enough repeated Python
  boilerplate to justify adding one. If launch-file helpers start
  duplicating, add a `genie_sim_common/` Python package here and
  install it via `ament_python_install_package(genie_sim_common)`.
- **Runtime deps.** This package has none — `package.xml` declares
  `ament_cmake_core` as a build tool and an exported build tool.
  Consumers declare their own dependency on `genie_sim_common`.
