# genie_sim_common

Shared CMake helpers for `genie_sim_*` packages in this
workspace. Its only purpose is to hold the conventions (build type,
C++ standard, warning flags, test wrappers) in one place so every
package's `CMakeLists.txt` starts with the same two lines instead of
copy-pasted preambles.

## Usage

Add the dep in `package.xml`:

```xml
<depend>genie_sim_common</depend>
```

Call the macro at the top of `CMakeLists.txt`, before any other
`find_package(...)`:

```cmake
find_package(genie_sim_common REQUIRED)
genie_sim_package()          # C++20 default
# or
genie_sim_package(WERROR)    # opt-in warnings-as-errors
```

That call sets:

- `CMAKE_BUILD_TYPE=Release` (when not otherwise set)
- `CMAKE_CXX_STANDARD=20`, required, no extensions
- `CMAKE_POSITION_INDEPENDENT_CODE=ON`
- `CMAKE_EXPORT_COMPILE_COMMANDS=ON`
- `-Wall -Wextra -Wpedantic` on gcc/clang (plus `-Werror` when `WERROR`)
- an opt-in `GENIE_SIM_COVERAGE_ENABLED` gate

Test wrappers `genie_sim_add_{test,gtest,gmock,pytest_test}` prefer
`ament_add_ros_isolated_*` when `USE_ISOLATED_TESTS=ON` and
the isolated variants are available (ROS 2 Kilted+). Otherwise they use
the corresponding `ament_add_*` function.

## Implementation

The [CMake helpers](cmake/) export macros through CONFIG_EXTRAS so downstream
packages need only `find_package(genie_sim_common REQUIRED)`. This wiring is
modelled on [`nav2_common`](https://github.com/ros-navigation/navigation2/tree/main/nav2_common).
