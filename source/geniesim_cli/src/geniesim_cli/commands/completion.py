# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from __future__ import annotations

import sys

from geniesim_cli._style import BOLD, CYAN, DIM, GREEN, RED, RST, YELLOW

_BASH_SCRIPT = r"""# geniesim bash completion — eval "$(geniesim completion bash)"
_geniesim_completions() {
    local cur prev words
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"
    words=("${COMP_WORDS[@]}")

    local cmds="ros deploy benchmark teleop autocollect docker docker6.2 docker4.5 docker5.1 status doctor env bootstrap tool dataset version completion"

    case "${#words[@]}" in
        2)
            COMPREPLY=($(compgen -W "${cmds}" -- "${cur}"))
            return ;;
    esac

    case "${words[1]}" in
        ros)
            case "${#words[@]}" in
                3) COMPREPLY=($(compgen -W "build doctor" -- "${cur}")) ;;
                4)
                    if [[ "${words[2]}" == "build" ]]; then
                        COMPREPLY=($(compgen -W "dev release cleanup" -- "${cur}"))
                    fi ;;
            esac ;;
        deploy)
            if [[ "${#words[@]}" -eq 3 ]]; then
                COMPREPLY=($(compgen -W "list geniesim geniesim_cli geniesim_benchmark geniesim_generator geniesim_ros geniesim_teleop" -- "${cur}"))
            fi ;;
        docker6.2|docker|docker4.5|docker5.1)
            if [[ "${#words[@]}" -eq 3 ]]; then
                COMPREPLY=($(compgen -W "build up down into logs" -- "${cur}"))
            fi ;;
        completion)
            if [[ "${#words[@]}" -eq 3 ]]; then
                COMPREPLY=($(compgen -W "bash zsh" -- "${cur}"))
            fi ;;
        tool)
            if [[ "${#words[@]}" -eq 3 ]]; then
                COMPREPLY=($(compgen -W "deps-dag ros-dag docs" -- "${cur}"))
            fi ;;
        dataset)
            case "${#words[@]}" in
                3) COMPREPLY=($(compgen -W "convert" -- "${cur}")) ;;
                4)
                    if [[ "${words[2]}" == "convert" ]]; then
                        COMPREPLY=($(compgen -W "agibot-to-lerobot" -- "${cur}"))
                    fi ;;
            esac ;;
    esac
}
complete -F _geniesim_completions geniesim
"""

_ZSH_SCRIPT = r"""#compdef geniesim
# geniesim zsh completion — eval "$(geniesim completion zsh)"

_geniesim() {
    local -a commands
    commands=(
        'ros:ROS 2 colcon build + rosdep doctor'
        'deploy:Build wheels into ./deploy'
        'benchmark:Run / list / batch benchmark tasks'
        'teleop:VR / Pico teleoperation (W.I.P.)'
        'autocollect:Auto-collect benchmark trajectories'
        'docker:Manage the GenieSim container (default → docker5.1)'
        'docker6.2:Isaac Sim 6.2 variant (geniesim4) — behavior unverified'
        'docker4.5:Isaac Sim 4.5 variant (geniesim2 E.O.L.)'
        'docker5.1:Isaac Sim 5.1 variant (geniesim3)'
        'status:Health-check all distributions'
        'doctor:Diagnose and repair'
        'env:Show GENIESIM_* env vars'
        'bootstrap:Bootstrap the geniesim stack'
        'tool:Contributor repo-maintenance utilities'
        'dataset:Dataset format conversion / inspection'
        'version:Show version information'
        'completion:Generate shell completion script'
    )

    local -a ros_cmds=(
        'build:Build colcon workspace'
        'doctor:Check & fix rosdep dependencies'
    )
    local -a ros_build_profiles=('dev' 'release' 'cleanup')
    local -a deploy_targets=('list' 'geniesim' 'geniesim_cli' 'geniesim_benchmark' 'geniesim_generator' 'geniesim_ros' 'geniesim_teleop')
    local -a docker_cmds=('build' 'up' 'down' 'into' 'logs')
    local -a tool_cmds=('deps-dag' 'ros-dag' 'docs')
    local -a dataset_cmds=('convert')
    local -a dataset_convert_pairs=('agibot-to-lerobot')
    local -a completion_shells=('bash' 'zsh')

    if (( CURRENT == 2 )); then
        _describe 'command' commands
        return
    fi

    case "${words[2]}" in
        ros)
            if (( CURRENT == 3 )); then
                _describe 'subcommand' ros_cmds
            elif (( CURRENT == 4 )) && [[ "${words[3]}" == "build" ]]; then
                _values 'profile' ${ros_build_profiles[@]}
            fi ;;
        deploy)
            if (( CURRENT == 3 )); then
                _values 'target' ${deploy_targets[@]}
            fi ;;
        docker6.2|docker|docker4.5|docker5.1)
            if (( CURRENT == 3 )); then
                _values 'action' ${docker_cmds[@]}
            fi ;;
        completion)
            if (( CURRENT == 3 )); then
                _values 'shell' ${completion_shells[@]}
            fi ;;
        tool)
            if (( CURRENT == 3 )); then
                _values 'subcommand' ${tool_cmds[@]}
            fi ;;
        dataset)
            if (( CURRENT == 3 )); then
                _values 'subcommand' ${dataset_cmds[@]}
            elif (( CURRENT == 4 )) && [[ "${words[3]}" == "convert" ]]; then
                _values 'pair' ${dataset_convert_pairs[@]}
            fi ;;
    esac
}

_geniesim "$@"
"""


def run(args: list[str]) -> None:
    if not args or args[0] in ("-h", "--help"):
        print(f"{BOLD}{CYAN}🐚 geniesim completion{RST}")
        print()
        print(f"{BOLD}Usage:{RST} geniesim completion {CYAN}<shell>{RST}")
        print()
        print(f"{BOLD}Shells:{RST}")
        print(f'  {CYAN}bash{RST}  {DIM}— add to ~/.bashrc:{RST}  eval "$(geniesim completion bash)"')
        print(f'  {CYAN}zsh{RST}   {DIM}— add to ~/.zshrc:{RST}   eval "$(geniesim completion zsh)"')
        sys.exit(0)

    shell = args[0]
    if shell == "bash":
        print(_BASH_SCRIPT)
    elif shell == "zsh":
        print(_ZSH_SCRIPT)
    else:
        print(f"{RED}❌ Unknown shell '{shell}'. Supported: bash, zsh{RST}")
        sys.exit(1)
