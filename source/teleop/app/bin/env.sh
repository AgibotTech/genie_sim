if [ -n "$BASH_SOURCE" ]; then
    SCRIPT_PATH="$BASH_SOURCE"
    SHELL_NAME="bash"
elif [ -n "$ZSH_VERSION" ]; then
    SCRIPT_PATH="${(%):-%x}"
    SHELL_NAME="zsh"
else
    SCRIPT_PATH="$0"
    SHELL_NAME=$(basename "$SHELL")
fi

SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
INSTALL_DIR=$(dirname "$SCRIPT_DIR")

# echo "Script dir: $SCRIPT_DIR"
# echo "Install dir: $INSTALL_DIR"

# ============================ utils ================================
is_in_container() {
    # common container indicators
    [ -f /.dockerenv ] && return 0
    [ -f /run/.containerenv ] && return 0
    grep -qaE '(docker|containerd|kubepods|libpod|lxc)' /proc/1/cgroup 2>/dev/null && return 0
    [ "${container:-}" = "docker" ] && return 0
    return 1
}
# ============================ ros ===============================
setup_ros_env() {
    local DEFAULT_ROS_DISTRO="humble"
    # Check if ROS_DISTRO environment variable exists
    if [ -z "${ROS_DISTRO}" ]; then
        # ROS environment not sourced, check if setup.bash exists
        if [ -f "/opt/ros/${DEFAULT_ROS_DISTRO}/setup.bash" ]; then
            source /opt/ros/${DEFAULT_ROS_DISTRO}/setup.${SHELL_NAME}
            echo "source /opt/ros/${DEFAULT_ROS_DISTRO}/setup.${SHELL_NAME}"
        else
            echo "Warning: /opt/ros/${DEFAULT_ROS_DISTRO}/setup.${SHELL_NAME} not found"
        fi
    else
        echo "ROS environment already sourced (ROS_DISTRO=$ROS_DISTRO)"
    fi

    # source all local_setup.${SHELL_NAME} under ${INSTALL_DIR}/share
    for dir in "${INSTALL_DIR}/share"/*; do
        if [ -d "$dir" ] && [ -f "$dir/local_setup.${SHELL_NAME}" ]; then
            source "$dir/local_setup.${SHELL_NAME}"
            echo "source $dir/local_setup.${SHELL_NAME}"
        fi
    done

    # if USE_CONTAINER is set, use ROS_LOCALHOST_ONLY=0; otherwise 1
    if [ -n "$USE_CONTAINER" ]; then
        if is_in_container; then
            export ROS_NET_IFACE=$(awk -v h="$(hostname)" '$2==h{print $1; exit}' /etc/hosts)
        else
            export ROS_NET_IFACE="xxx.xx.x.x" # default docker bridge ip
        fi
        echo "Set FASTRTPS interfaceWhiteList Address: $ROS_NET_IFACE"
        envsubst < $SCRIPT_DIR/fastdds_server.xml.in > /tmp/fastdds_server.xml
        export FASTRTPS_DEFAULT_PROFILES_FILE="/tmp/fastdds_server.xml"
        echo "Set FASTRTPS_DEFAULT_PROFILES_FILE to $FASTRTPS_DEFAULT_PROFILES_FILE"
        export ROS_LOCALHOST_ONLY=0
        export ROS_DOMAIN_ID=1
        echo "export ROS_LOCALHOST_ONLY=0"
    else
        export ROS_LOCALHOST_ONLY=1
        # export ROS_DOMAIN_ID=0
        echo "export ROS_LOCALHOST_ONLY=1"
    fi
}
echo "==================== ENV SETUP START ===================="
setup_ros_env
echo "==================== ENV SETUP END ===================="
