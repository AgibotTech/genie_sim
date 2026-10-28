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

    local cmds="ros deploy benchmark teleop post-train data-collection docker docker6.2 docker4.5 docker5.1 status doctor env bootstrap tool dataset version completion"

    case "${#words[@]}" in
        2)
            COMPREPLY=($(compgen -W "${cmds}" -- "${cur}"))
            return ;;
    esac

    case "${words[1]}" in
        bootstrap)
            COMPREPLY=($(compgen -W "--help" -- "${cur}")) ;;
        ros)
            case "${#words[@]}" in
                3) COMPREPLY=($(compgen -W "build doctor" -- "${cur}")) ;;
                4)
                    if [[ "${words[2]}" == "build" ]]; then
                        COMPREPLY=($(compgen -W "dev release cleanup" -- "${cur}"))
                    fi ;;
            esac ;;
        data-collection)
            if [[ "${#words[@]}" -eq 3 ]]; then
                COMPREPLY=($(compgen -W "install config llm replay list tasks robots run build up into down" -- "${cur}"))
            elif [[ "${#words[@]}" -eq 4 && "${words[2]}" == "llm" ]]; then
                COMPREPLY=($(compgen -W "list show run" -- "${cur}"))
            fi ;;
        deploy)
            if [[ "${#words[@]}" -eq 3 ]]; then
                COMPREPLY=($(compgen -W "list geniesim geniesim_cli geniesim_benchmark geniesim_scene_agent geniesim_ros geniesim_teleop" -- "${cur}"))
            fi ;;
        docker6.2|docker|docker4.5|docker5.1)
            if [[ "${words[1]}" == docker && "${words[2]}" == --profile ]]; then
                case "$COMP_CWORD" in
                    3) COMPREPLY=($(compgen -W "post-train post-train-teleop" -- "${cur}")) ;;
                    4) COMPREPLY=($(compgen -W "build up down into logs" -- "${cur}")) ;;
                    5)
                        case "${words[4]}" in
                            build) COMPREPLY=($(compgen -W "--china --build-arg --no-cache --progress" -- "${cur}")) ;;
                            up) COMPREPLY=($(compgen -W "--headless" -- "${cur}")) ;;
                            logs) COMPREPLY=($(compgen -W "--follow" -- "${cur}")) ;;
                        esac ;;
                esac
            elif [[ "${#words[@]}" -eq 3 ]]; then
                local actions="build up down into logs"
                [[ "${words[1]}" == docker ]] && actions+=" --profile"
                COMPREPLY=($(compgen -W "${actions}" -- "${cur}"))
            fi ;;
        post-train)
            if (( COMP_CWORD == 2 )); then
                COMPREPLY=($(compgen -W "server teleop" -- "${cur}"))
            elif [[ "${words[2]}" == server ]]; then
                if (( COMP_CWORD == 3 )); then
                    COMPREPLY=($(compgen -W "start status stop logs" -- "${cur}"))
                elif (( COMP_CWORD == 4 )); then
                    case "${words[3]}" in
                        start|status|stop)
                            COMPREPLY=($(compgen -W "--count --base-port --devices --only --dry-run --code --assets --no-fabric --rebuild-ws --barrier-cap-s --help" -- "${cur}")) ;;
                        logs) COMPREPLY=($(compgen -W "--port --follow --help" -- "${cur}")) ;;
                    esac
                fi
            elif [[ "${words[2]}" == teleop ]] && (( COMP_CWORD == 3 )); then
                COMPREPLY=($(compgen -W "--addr --task --instance --port --viewer-port --host_ip --help" -- "${cur}"))
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
        'post-train:Post-training simulation servers and Pico debugging'
        'data-collection:Collect LLM/JSON tasks and replay recorded episodes'
        'docker:Manage the GenieSim container (default → docker5.1)'
        'docker6.2:Isaac Sim 6.2 variant (geniesim4) — behaviour unverified'
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
    local -a deploy_targets=('list' 'geniesim' 'geniesim_cli' 'geniesim_benchmark' 'geniesim_scene_agent' 'geniesim_ros' 'geniesim_teleop')
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
        bootstrap)
            _values 'option' --help ;;
        ros)
            if (( CURRENT == 3 )); then
                _describe 'subcommand' ros_cmds
            elif (( CURRENT == 4 )) && [[ "${words[3]}" == "build" ]]; then
                _values 'profile' ${ros_build_profiles[@]}
            fi ;;
        data-collection)
            if (( CURRENT == 3 )); then
                _values 'subcommand' install config llm replay list tasks robots run build up into down
            elif (( CURRENT == 4 )) && [[ "${words[3]}" == "llm" ]]; then
                _values 'action' list show run
            fi ;;
        deploy)
            if (( CURRENT == 3 )); then
                _values 'target' ${deploy_targets[@]}
            fi ;;
        docker6.2|docker|docker4.5|docker5.1)
            if [[ "${words[2]}" == docker && "${words[3]}" == --profile ]]; then
                case "$CURRENT" in
                    4) _values 'profile' post-train post-train-teleop ;;
                    5) _values 'action' ${docker_cmds[@]} ;;
                    6)
                        case "${words[5]}" in
                            build) _values 'option' --china --build-arg --no-cache --progress ;;
                            up) _values 'option' --headless ;;
                            logs) _values 'option' --follow ;;
                        esac ;;
                esac
            elif (( CURRENT == 3 )); then
                if [[ "${words[2]}" == docker ]]; then
                    _values 'action' ${docker_cmds[@]} '--profile[Post-training image profile]'
                else
                    _values 'action' ${docker_cmds[@]}
                fi
            fi ;;
        post-train)
            if (( CURRENT == 3 )); then
                _values 'command' server teleop
            elif [[ "${words[3]}" == server ]]; then
                if (( CURRENT == 4 )); then
                    _values 'action' start status stop logs
                elif (( CURRENT == 5 )); then
                    case "${words[4]}" in
                        start|status|stop)
                            _values 'option' --count --base-port --devices --only --dry-run --code --assets --no-fabric --rebuild-ws --barrier-cap-s --help ;;
                        logs) _values 'option' --port --follow --help ;;
                    esac
                fi
            elif [[ "${words[3]}" == teleop ]] && (( CURRENT == 4 )); then
                _values 'option' --addr --task --instance --port --viewer-port --host_ip --help
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
