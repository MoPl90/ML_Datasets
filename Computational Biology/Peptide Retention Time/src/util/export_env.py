import re
import subprocess
import sys
import argparse

import yaml


def export_env(name, history_only=False, include_builds=False):
    """Capture `conda env export` output"""
    cmd = ["conda", "env", "export", "--name", name]
    if history_only:
        cmd.append("--from-history")
        if include_builds:
            raise ValueError('Cannot include build versions with "from history" mode')
    if not include_builds:
        cmd.append("--no-builds")
    cp = subprocess.run(" ".join(cmd), stdout=subprocess.PIPE, shell=True)
    try:
        cp.check_returncode()
    except:
        raise
    else:
        return yaml.safe_load(cp.stdout)


def _is_history_dep(d, history_deps):
    if not isinstance(d, str):
        return False
    d_prefix = re.sub(r"=.*", "", d)
    return d_prefix in history_deps


def _get_pip_deps(full_deps):
    for dep in full_deps:
        if isinstance(dep, dict) and "pip" in dep:
            return dep


def _combine_env_data(env_data_full, env_data_hist):
    deps_full = env_data_full["dependencies"]
    deps_hist = [h.split("=")[0] for h in env_data_hist["dependencies"]]
    deps = [dep for dep in deps_full if _is_history_dep(dep, deps_hist)]

    pip_deps = _get_pip_deps(deps_full)

    env_data = {}
    env_data["name"] = env_data_full["name"]
    env_data["channels"] = env_data_full["channels"]
    env_data["dependencies"] = deps
    env_data["dependencies"].append(pip_deps)

    return env_data


def main(name, file):
    env_data_full = export_env(name=name)
    env_data_hist = export_env(name=name, history_only=True)
    env_data = _combine_env_data(env_data_full, env_data_hist)
    with open(file, "w", encoding="utf-8") as yaml_file:
        yaml.dump(env_data, yaml_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Exports Conda Environment Including Pip Libraries."
    )
    parser.add_argument("--env_name", help="name of the conda environment")
    parser.add_argument("--file", default="environment.yml", help="export file name")
    args = parser.parse_args()

    main(name=args.env_name, file=args.file)
