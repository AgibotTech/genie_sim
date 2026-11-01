# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0
# Detect source-tree, file-list, build-rule, and native-code changes.
RLINF_WS_PKGS_DEFAULT="genie_sim_common genie_sim_engine_core genie_sim_engine_isaacsim genie_sim_engine_newton genie_sim_render"

ws_stamp_parts() {
  local src_root="$1"; shift
  local pkgs="$*"
  local p real
  real="$(cd "$src_root" 2>/dev/null && pwd -P)" || real="$src_root"
  echo "root:${real}"

  {
    for p in $pkgs; do
      [ -d "$src_root/$p" ] || { echo "MISSING $p"; continue; }
      find -L "$src_root/$p" -type f \
        ! -path '*/__pycache__/*' ! -name '*.pyc' ! -name '*.pyo' \
        2>/dev/null | sed "s#^$src_root/##" | LC_ALL=C sort
    done
  } | sha256sum | cut -d' ' -f1 | sed 's/^/files:/'

  {
    for p in $pkgs; do
      find -L "$src_root/$p" -type f \
        \( -name CMakeLists.txt -o -name package.xml -o -name '*.cmake' \
           -o -name 'setup.py' -o -name 'setup.cfg' \) \
        2>/dev/null | LC_ALL=C sort | xargs -r cat 2>/dev/null
    done
  } | sha256sum | cut -d' ' -f1 | sed 's/^/build:/'

  {
    for p in $pkgs; do
      find -L "$src_root/$p" -type f \
        \( -name '*.c' -o -name '*.cc' -o -name '*.cpp' -o -name '*.cxx' \
           -o -name '*.h' -o -name '*.hpp' -o -name '*.hh' \) \
        2>/dev/null | LC_ALL=C sort | xargs -r cat 2>/dev/null
    done
  } | sha256sum | cut -d' ' -f1 | sed 's/^/native:/'
}

ws_stamp_write() {
  local ws_build="$1" src_root="$2"; shift 2
  mkdir -p "$ws_build/install" 2>/dev/null
  ws_stamp_parts "$src_root" "$@" > "$ws_build/install/.rlinf_build_stamp"
}

ws_stamp_reason() {
  local ws_build="$1" src_root="$2"; shift 2
  local f="$ws_build/install/.rlinf_build_stamp"
  [ -f "$f" ] || { echo "Build stamp missing; cannot verify installed workspace"; return; }
  local now old k a b
  now="$(ws_stamp_parts "$src_root" "$@")"
  for k in root files build native; do
    a="$(printf '%s\n' "$now" | grep "^$k:" | head -1)"
    b="$(grep "^$k:" "$f" | head -1)"
    [ "$a" = "$b" ] && continue
    case "$k" in
      root)   echo "Source root changed: built=${b#root:}  current=${a#root:}" ;;
      files)  echo "File list changed; symlink-install must register added or removed files" ;;
      build)  echo "Build rules changed (CMakeLists / package.xml / *.cmake)" ;;
      native) echo "C/C++ source changed; native artifacts need rebuilding" ;;
    esac
  done
}

# Call while holding the workspace lock. Package selection remains configurable
# for runtime images that install only a subset of the available backends.
ws_prepare() {
  local src="$1" build="$2" force="${3:-0}" reason="" package
  local packages="${RLINF_WS_PACKAGES:-$RLINF_WS_PKGS_DEFAULT}"
  [ -d "$src" ] || { echo "ROS workspace source missing: $src" >&2; return 1; }
  for package in $packages; do
    [[ "$package" =~ ^[a-zA-Z0-9_]+$ ]] && [ -d "$src/$package" ] || {
      echo "ROS package missing or invalid: $package" >&2; return 1;
    }
  done
  if [ "$force" != 1 ] && [ -f "$build/install/setup.bash" ]; then
    reason="$(ws_stamp_reason "$build" "$src" $packages)"
    [ -n "$reason" ] || return 0
    printf 'ROS workspace needs rebuild: %s\n' "$reason" >&2
    if [ "${RLINF_NO_AUTO_REBUILD:-0}" = 1 ]; then
      echo "RLINF_NO_AUTO_REBUILD=1: retaining the existing workspace" >&2
      return 0
    fi
  fi
  slot_require_idle_build "${RLINF_RUN_ROOT:-/tmp}" || return 1
  if [ "$force" = 1 ]; then
    local resolved source_root
    resolved="$(realpath -m "$build")"
    source_root="$(realpath -m "$src")"
    case "$resolved" in /|/tmp|/home|/usr|/opt|/workspace|"${HOME:-/}")
      echo "Unsafe workspace build root: $resolved" >&2; return 1;;
    esac
    case "$source_root/" in "$resolved/"*)
      echo "Build root must not contain source: $resolved" >&2; return 1;;
    esac
    # Only remove artifacts of a workspace this helper previously recorded.
    if [ -f "$build/install/.rlinf_build_stamp" ]; then
      rm -rf -- "$build/build" "$build/install" "$build/log"
    fi
  fi
  mkdir -p "$build/src" || return 1
  for package in $packages; do
    ln -sfn "$src/$package" "$build/src/$package" || return 1
  done
  echo "Building ROS workspace ($packages): $src -> $build"
  (
    set -e
    export PATH=/usr/bin:/bin:/usr/sbin:/sbin
    unset PYTHONPATH CONDA_PREFIX
    source "/opt/ros/${ROS_DISTRO:-jazzy}/setup.bash"
    cd "$build"
    colcon build --symlink-install --packages-select $packages \
      --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3 -DPYTHON_EXECUTABLE=/usr/bin/python3
  ) || return 1
  ws_stamp_write "$build" "$src" $packages
}
