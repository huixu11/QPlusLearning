"""Build causal-conv1d 1.7.0 locally only for an incompatible prebuilt GLIBC.

No system libraries, Torch dependencies, project locks or training data are
changed. The CLI repairs one already-owned laboratory training interpreter.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
from urllib.request import urlopen


SOURCE_URL = "https://files.pythonhosted.org/packages/f4/d1/eba27735f31bd5527d39a3c7f693e6ded7b844cab275db719a15ad37b6cd/causal_conv1d-1.7.0.tar.gz"
SOURCE_SHA256 = "3202758494eaa7b597ce1c282dfa188889506bfcb92cad3c407d26736bfcd32b"
BUILD_TOOLS = {"setuptools": "80.9.0", "wheel": "0.45.1", "packaging": "25.0"}
PLATFORM_PREFIX = "QPLUS_NATIVE_PLATFORM "
VERIFIED_PREFIX = "QPLUS_NATIVE_VERIFIED "
PLATFORM_CODE = r'''
import importlib.metadata as md, json, os, platform, shutil, sys, torch
from pathlib import Path
# Do not import torch.utils.cpp_extension here: optional build tooling may be
# absent, and its import would hide the native extension's original GLIBC error.
CUDA_HOME = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
if not CUDA_HOME:
    nvcc = shutil.which('nvcc')
    if nvcc:
        CUDA_HOME = str(Path(nvcc).resolve().parent.parent)
tools = {}
for package in ('setuptools', 'wheel', 'packaging'):
    try: tools[package] = md.version(package)
    except md.PackageNotFoundError: tools[package] = None
print('QPLUS_NATIVE_PLATFORM ' + json.dumps({
    'python': f'{sys.version_info.major}.{sys.version_info.minor}',
    'prefix': sys.prefix, 'system': platform.system(), 'machine': platform.machine(),
    'libc': platform.libc_ver(), 'torch': torch.__version__, 'cuda': torch.version.cuda,
    'cxx11_abi': torch._C._GLIBCXX_USE_CXX11_ABI, 'cuda_home': CUDA_HOME,
    'build_tools': tools}), flush=True)
'''
IMPORT_CODE = PLATFORM_CODE + r'''
import causal_conv1d, causal_conv1d_cuda
sys.path.insert(0, sys.argv[1])
from optimized_training import require_optimized_bindings
result = require_optimized_bindings()
print('QPLUS_NATIVE_VERIFIED ' + json.dumps(result), flush=True)
'''


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def glibc_mismatch(diagnostic):
    return re.search(r"GLIBC_\d+\.\d+[^\n]*not found", diagnostic) is not None


def _json_line(output, prefix):
    for line in output.splitlines():
        if line.startswith(prefix):
            return json.loads(line[len(prefix):])
    return None


def _verified_bindings(output):
    bindings = _json_line(output, VERIFIED_PREFIX)
    kernels = bindings.get("kernels", {}) if isinstance(bindings, dict) else {}
    if (not kernels.get("convolution", "").startswith("causal_conv1d.")
            or not kernels.get("delta_rule", "").startswith("fla.")):
        raise RuntimeError("Training interpreter did not return verified optimized convolution and FLA bindings")
    return bindings


def _target_environment(python, env):
    python = Path(python).expanduser().absolute()  # Preserve a venv interpreter symlink's prefix.
    if python.parent.name != "bin" or not python.name.startswith("python"):
        raise ValueError("Use the laboratory training prefix's bin/python")
    prefix = python.parent.parent.resolve()
    if not (prefix / "pyvenv.cfg").is_file():
        raise ValueError("Native repair requires an existing laboratory uv venv, not a base interpreter")
    if any((parent / "conda-meta").exists() for parent in (prefix, *prefix.parents)):
        raise ValueError("Native repair must not install into a Conda or Jupyter environment")
    ownership = prefix.with_name("." + prefix.name + ".qplus-kev-runtime.json")
    try:
        receipt = json.loads(ownership.read_text())
    except (OSError, ValueError):
        raise ValueError("Training prefix has no valid LabRuntime ownership receipt") from None
    if (receipt.get("format") != "qplus-kev-uv-environment-v1"
            or receipt.get("training_environment") != str(prefix)):
        raise ValueError("Training prefix ownership differs from this repair target")
    scoped = dict(env)
    scoped["UV_PROJECT_ENVIRONMENT"] = str(prefix)
    scoped["VIRTUAL_ENV"] = str(prefix)
    for name in ("NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS", "CAUSAL_CONV1D_LOCAL_VERSION"):
        scoped.pop(name, None)
    return python, prefix, scoped


def _run(command, repo, env, **kwargs):
    return subprocess.run([str(value) for value in command], cwd=repo, env=dict(env), **kwargs)


def _probe(python, repo, env):
    result = _run([python, "-u", "-c", IMPORT_CODE, Path(__file__).resolve().parent],
                  repo, env, capture_output=True, text=True)
    return result, _json_line(result.stdout, PLATFORM_PREFIX)


def _validate_platform(info, prefix):
    if info is None:
        raise RuntimeError("Could not inspect the training Torch environment; this repair only rebuilds causal-conv1d")
    expected = {"python": "3.13", "system": "Linux", "machine": "x86_64", "cuda": "12.8", "cxx11_abi": True}
    for name, value in expected.items():
        if info.get(name) != value:
            raise RuntimeError(f"Local kernel build requires {name}={value!r}; received {info.get(name)!r}")
    if not info.get("torch", "").split("+")[0] == "2.8.0" or Path(info["prefix"]).resolve() != prefix:
        raise RuntimeError("Local kernel build requires pinned Torch 2.8.0 in the owned training prefix")


def _toolchain(info, repo, env):
    cuda_home = env.get("CUDA_HOME") or info.get("cuda_home")
    if not cuda_home:
        raise RuntimeError("A CUDA 12.8 developer toolkit with nvcc is required. Set CUDA_HOME and PATH; "
                           "nvidia-smi's CUDA version does not establish that nvcc is installed.")
    cuda_home = Path(cuda_home).expanduser().resolve()
    nvcc = cuda_home / "bin/nvcc"
    if not nvcc.is_file():
        raise RuntimeError(f"CUDA_HOME/bin/nvcc is missing: {nvcc}. Set CUDA_HOME to a CUDA 12.8 developer toolkit.")
    commands = {"nvcc": [str(nvcc), "--version"]}
    for name, variable, default in (("cc", "CC", "gcc"), ("cxx", "CXX", "g++")):
        arguments = shlex.split(env.get(variable) or default)
        executable = shutil.which(arguments[0], path=env.get("PATH")) if arguments else None
        if not executable:
            raise RuntimeError(f"Missing native-build tool {name}; configure a working compiler with {variable}")
        # A CLI's /usr/bin/gcc and a notebook's PATH-selected gcc should share
        # the cache only when they resolve to the same actual compiler.
        commands[name] = [str(Path(executable).resolve()), *arguments[1:], "--version"]
    metadata = {"cuda_home": str(cuda_home), "official_architectures": "setup.py sm75/sm80/sm87/sm90/sm100/sm120"}
    for name, command in commands.items():
        if not command or not shutil.which(command[0], path=env.get("PATH")):
            raise RuntimeError(f"Missing native-build tool {name}; provide CUDA 12.8 nvcc and working C/C++ compilers using CUDA_HOME, CC and CXX")
        result = _run(command, repo, env, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(f"Native-build tool {name} failed: {result.stderr or result.stdout}")
        metadata[name] = {"command": command, "version": result.stdout.strip()}
    version = re.search(r"release\s+(\d+)\.(\d+)", metadata["nvcc"]["version"])
    if version is None or tuple(map(int, version.groups())) != (12, 8):
        raise RuntimeError("Pinned Torch uses CUDA 12.8; local compilation requires nvcc 12.8. "
                           f"Received {metadata['nvcc']['version']}. Set CUDA_HOME and PATH to the CUDA 12.8 developer toolkit.")
    return metadata


def _build_environment(env, toolchain):
    scoped = dict(env)
    cuda_home = Path(toolchain["cuda_home"])
    includes = [path for path in (cuda_home / "targets/x86_64-linux/include", cuda_home / "include") if path.is_dir()]
    libraries = [path for path in (cuda_home / "targets/x86_64-linux/lib", cuda_home / "lib64", cuda_home / "lib") if path.is_dir()]
    if not any((path / "cuda_bf16.h").is_file() for path in includes):
        raise RuntimeError("CUDA 12.8 developer headers are missing; install cuda-cudart-dev and cuda-cccl in the toolkit prefix")
    for name, paths in (("CPATH", includes), ("LIBRARY_PATH", libraries)):
        entries = [str(path) for path in paths]
        if scoped.get(name):
            entries.append(scoped[name])
        scoped[name] = os.pathsep.join(entries)
    scoped.update(CUDA_HOME=str(cuda_home), MAX_JOBS="2",
                  CAUSAL_CONV1D_FORCE_BUILD="TRUE", CAUSAL_CONV1D_SKIP_CUDA_BUILD="FALSE",
                  CAUSAL_CONV1D_FORCE_CXX11_ABI="TRUE")
    scoped["CC"] = shlex.join(toolchain["cc"]["command"][:-1])
    scoped["CXX"] = shlex.join(toolchain["cxx"]["command"][:-1])
    # Platform validation has already established that Torch's actual ABI is
    # True. Remove inherited overrides that could change this pinned build.
    for name in ("CAUSAL_CONV1D_LOCAL_VERSION", "NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS"):
        scoped.pop(name, None)
    return scoped


def _verified_source(cache):
    archive = cache / "causal_conv1d-1.7.0.tar.gz"
    if not archive.exists():
        with tempfile.NamedTemporaryFile(dir=cache, suffix=".download", delete=False) as target:
            temporary = Path(target.name)
            try:
                with urlopen(SOURCE_URL, timeout=60) as source:
                    shutil.copyfileobj(source, target)
                target.flush()
                if file_sha256(temporary) != SOURCE_SHA256:
                    raise ValueError("Official causal-conv1d source checksum mismatch")
                temporary.replace(archive)
            finally:
                temporary.unlink(missing_ok=True)
    if file_sha256(archive) != SOURCE_SHA256:
        raise ValueError("Cached causal-conv1d source checksum mismatch; preserve it and use a fresh native cache")
    return archive


def _extract_source(archive, destination):
    with tarfile.open(archive, "r:gz") as source:
        for member in source.getmembers():
            name = Path(member.name)
            if (name.is_absolute() or ".." in name.parts or not name.parts
                    or name.parts[0] != "causal_conv1d-1.7.0" or member.issym() or member.islnk()):
                raise ValueError("Unsafe path in causal-conv1d source archive")
        source.extractall(destination, filter="data")
    return destination / "causal_conv1d-1.7.0"


def _cache_wheel(directory, fingerprint):
    receipt_file = directory / "receipt.json"
    if not receipt_file.exists():
        return None
    receipt = json.loads(receipt_file.read_text())
    if receipt.get("fingerprint") != fingerprint or receipt.get("verified_import") is not True:
        raise ValueError("Native wheel receipt does not match this source/platform/toolchain")
    filename = receipt.get("wheel", "")
    if not filename or Path(filename).name != filename or not filename.endswith(".whl"):
        raise ValueError("Invalid native wheel path in cache receipt")
    wheel = directory / filename
    if not wheel.is_file() or file_sha256(wheel) != receipt.get("wheel_sha256"):
        raise ValueError("Cached native wheel checksum mismatch; preserve it and use a fresh native cache")
    return wheel


def ensure_native_kernels(python, repo, workspace, env):
    """Verify native imports; rebuild only the same pinned package for GLIBC errors."""
    python, prefix, scoped = _target_environment(python, env)
    repo = Path(repo).resolve()
    workspace = Path(workspace).resolve()
    imported, info = _probe(python, repo, scoped)
    if imported.returncode == 0:
        _validate_platform(info, prefix)
        return {"status": "already_verified", "bindings": _verified_bindings(imported.stdout)}
    diagnostic = (imported.stdout or "") + "\n" + (imported.stderr or "")
    print("Native optimized-kernel import failed:\n" + diagnostic, flush=True)
    if not glibc_mismatch(diagnostic):
        raise RuntimeError("Native kernel failure is not a GLIBC-version mismatch; no automatic package replacement was attempted.\n" + diagnostic)
    _validate_platform(info, prefix)
    toolchain = _toolchain(info, repo, scoped)
    scoped = _build_environment(scoped, toolchain)
    # Keep all existing build-tool versions, including packages pinned by Kev.
    # Only missing tooling is added, with exact versions and no dependencies.
    missing = [f"{name}=={version}" for name, version in BUILD_TOOLS.items() if not info["build_tools"].get(name)]
    if missing:
        _run(["uv", "pip", "install", "--python", python, "--no-deps", *missing], repo, scoped, check=True)
        inspected = _run([python, "-u", "-c", PLATFORM_CODE], repo, scoped, capture_output=True, text=True, check=True)
        info = _json_line(inspected.stdout, PLATFORM_PREFIX)
        _validate_platform(info, prefix)
    fingerprint = {"source_url": SOURCE_URL, "source_sha256": SOURCE_SHA256,
                   "platform": info, "toolchain": toolchain,
                   "compile_environment": {name: scoped.get(name) for name in
                       ("CC", "CXX", "CFLAGS", "CXXFLAGS", "LDFLAGS", "CPATH", "LIBRARY_PATH")},
                   "max_jobs": 2, "force_build": True, "skip_cuda_build": False,
                   "force_cxx11_abi": True, "builder": "training-python setup.py bdist_wheel"}
    key = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()
    cache = workspace / "native-kernels"
    directory = cache / key
    directory.mkdir(parents=True, exist_ok=True)
    wheel = _cache_wheel(directory, fingerprint)
    built = wheel is None
    if built:
        archive = _verified_source(cache)
        print("Building checksum-verified causal-conv1d 1.7.0 against the local GLIBC/CUDA 12.8 toolchain. "
              "The official source builds several GPU architectures; compilation can take time.", flush=True)
        with tempfile.TemporaryDirectory(prefix="build-", dir=cache) as scratch:
            source = _extract_source(archive, Path(scratch))
            dist = Path(scratch) / "dist"
            # Use the exact training interpreter. bdist_wheel consumes installed
            # build tools without resolving dependencies or creating isolation;
            # FORCE_BUILD makes the upstream command bypass its wheel download.
            _run([python, "-u", "setup.py", "bdist_wheel", "--dist-dir", dist],
                 source, scoped, check=True)
            wheels = list(dist.glob("causal_conv1d-1.7.0-*.whl"))
            if len(wheels) != 1:
                raise RuntimeError("Local compilation did not produce exactly one causal-conv1d 1.7.0 wheel")
            wheel = directory / wheels[0].name
            shutil.copy2(wheels[0], wheel)
    _run(["uv", "pip", "install", "--python", python, "--no-deps", "--reinstall", wheel], repo, scoped, check=True)
    verified, verified_info = _probe(python, repo, scoped)
    if verified.returncode:
        raise RuntimeError("Locally compiled kernel failed native import or optimized binding verification:\n"
                           + (verified.stdout or "") + "\n" + (verified.stderr or ""))
    _validate_platform(verified_info, prefix)
    bindings = _verified_bindings(verified.stdout)
    receipt = {"fingerprint": fingerprint, "wheel": wheel.name,
               "wheel_sha256": file_sha256(wheel), "verified_import": True,
               "bindings": bindings}
    (directory / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return {"status": "built_and_verified" if built else "cached_and_verified",
            "wheel": str(wheel), "receipt": str(directory / "receipt.json"), "bindings": receipt["bindings"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", required=True, type=Path)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--workspace", required=True, type=Path)
    cli = parser.parse_args(argv)
    result = ensure_native_kernels(cli.python, cli.repo, cli.workspace, os.environ)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
