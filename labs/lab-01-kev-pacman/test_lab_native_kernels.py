"""CPU checks for safe native-build selection and cache integrity, not CUDA execution."""
import builtins
import contextlib
import io
from importlib.metadata import PackageNotFoundError
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import lab_native_kernels as native


def fixture(folder):
    root = Path(folder)
    prefix = root / "training"
    (prefix / "bin").mkdir(parents=True)
    (prefix / "pyvenv.cfg").write_text("home = fixture\n")
    python = prefix / "bin/python"
    python.write_text("fixture interpreter, never executed")
    prefix.with_name(".training.qplus-kev-runtime.json").write_text(json.dumps({
        "format": "qplus-kev-uv-environment-v1", "training_environment": str(prefix)}))
    toolkit = root / "cuda-toolkit"
    (toolkit / "bin").mkdir(parents=True)
    nvcc = toolkit / "bin/nvcc"
    nvcc.write_text("fixture compiler, never executed")
    nvcc.chmod(0o755)
    include = toolkit / "targets/x86_64-linux/include"
    include.mkdir(parents=True)
    (include / "cuda_bf16.h").write_text("fixture")
    (toolkit / "targets/x86_64-linux/lib").mkdir()
    repo = root / "kev"
    repo.mkdir()
    info = {"python": "3.13", "prefix": str(prefix), "system": "Linux", "machine": "x86_64",
            "libc": ["glibc", "2.31"], "torch": "2.8.0+cu128", "cuda": "12.8",
            "cxx11_abi": True, "cuda_home": str(toolkit), "build_tools": dict(native.BUILD_TOOLS)}
    env = {"PATH": os.environ["PATH"], "CUDA_HOME": str(toolkit), "CC": "gcc", "CXX": "g++",
           "CUDA_VISIBLE_DEVICES": "1", "UV_CACHE_DIR": str(root / "uv-cache"),
           "CPATH": "/existing/include", "LIBRARY_PATH": "/existing/lib"}
    return python, repo, root / "runtime", info, env


def probe_output(info, verified=False):
    output = native.PLATFORM_PREFIX + json.dumps(info) + "\n"
    if verified:
        output += native.VERIFIED_PREFIX + json.dumps({"kernels": {
            "convolution": "causal_conv1d.native", "delta_rule": "fla.native"}}) + "\n"
    return output


def toolchain_fixture(env):
    return {"cuda_home": env["CUDA_HOME"],
            "cc": {"command": [str(Path(shutil.which("gcc", path=env["PATH"])).resolve()), "--version"]},
            "cxx": {"command": [str(Path(shutil.which("g++", path=env["PATH"])).resolve()), "--version"]}}


class NativeKernelTests(unittest.TestCase):
    def test_platform_probe_reports_missing_build_tools_without_importing_cpp_extension(self):
        torch = SimpleNamespace(__version__="2.8.0+cu128", version=SimpleNamespace(cuda="12.8"),
                                _C=SimpleNamespace(_GLIBCXX_USE_CXX11_ABI=True))
        original_import = builtins.__import__
        def no_build_extension(name, *args, **kwargs):
            if name == "torch.utils.cpp_extension":
                raise AssertionError("Optional build tools must not hide the original native import error")
            return original_import(name, *args, **kwargs)
        output = io.StringIO()
        with patch.dict("sys.modules", {"torch": torch}), \
                patch.dict(os.environ, {"CUDA_HOME": "/explicit-cuda-12.8"}), \
                patch("builtins.__import__", side_effect=no_build_extension), \
                patch("importlib.metadata.version", side_effect=PackageNotFoundError), \
                contextlib.redirect_stdout(output):
            exec(native.PLATFORM_CODE, {})
        info = native._json_line(output.getvalue(), native.PLATFORM_PREFIX)
        self.assertEqual(info["cuda_home"], "/explicit-cuda-12.8")
        self.assertTrue(info["cxx11_abi"])
        self.assertEqual(info["build_tools"], {"setuptools": None, "wheel": None, "packaging": None})

    def test_only_glibc_version_not_found_selects_source_rebuild(self):
        self.assertTrue(native.glibc_mismatch("ImportError: libc.so.6: version `GLIBC_2.32' not found"))
        for error in ("GLIBCXX_3.4.32 not found", "undefined symbol: torch", "No module named causal_conv1d", "GLIBC_2.32 is available"):
            self.assertFalse(native.glibc_mismatch(error))

    def test_native_repair_requires_owned_uv_prefix_and_refuses_conda(self):
        with tempfile.TemporaryDirectory() as folder:
            python, repo, workspace, info, env = fixture(folder)
            _, prefix, scoped = native._target_environment(python, env)
            self.assertEqual(scoped["UV_PROJECT_ENVIRONMENT"], str(prefix))
            self.assertEqual(scoped["VIRTUAL_ENV"], str(prefix))
            self.assertNotIn("VIRTUAL_ENV", env)
            (prefix / "conda-meta").mkdir()
            with self.assertRaisesRegex(ValueError, "Conda or Jupyter"):
                native._target_environment(python, env)
        with tempfile.TemporaryDirectory() as folder:
            python, _, _, _, env = fixture(folder)
            python.parent.parent.with_name(".training.qplus-kev-runtime.json").unlink()
            with self.assertRaisesRegex(ValueError, "ownership"):
                native._target_environment(python, env)

    def test_wrong_cuda_nvcc_is_rejected_before_install_or_build(self):
        with tempfile.TemporaryDirectory() as folder:
            python, repo, _, info, env = fixture(folder)
            def version(command, repo, env, **kwargs):
                text = "Cuda compilation tools, release 11.7, V11.7.64" if command[0].endswith("nvcc") else "gcc 9.4.0"
                return subprocess.CompletedProcess(command, 0, text, "")
            with patch("lab_native_kernels._run", side_effect=version) as run:
                with self.assertRaisesRegex(RuntimeError, "requires nvcc 12.8"):
                    native._toolchain(info, repo, env)
                self.assertTrue(all(call.args[0][-1] == "--version" for call in run.call_args_list))

    def test_explicit_cuda_home_wins_over_platform_default_and_preserves_scoped_env(self):
        with tempfile.TemporaryDirectory() as folder:
            _, repo, _, info, env = fixture(folder)
            info["cuda_home"] = "/usr/local/cuda-11.7"
            def version(command, **kwargs):
                self.assertEqual(kwargs["env"], env)
                text = "Cuda compilation tools, release 12.8, V12.8.61" if command[0].endswith("nvcc") else "gcc 9.4.0"
                return subprocess.CompletedProcess(command, 0, text, "")
            with patch("lab_native_kernels.subprocess.run", side_effect=version) as run:
                toolchain = native._toolchain(info, repo, env)
            self.assertEqual(toolchain["cuda_home"], env["CUDA_HOME"])
            self.assertEqual(run.call_args_list[0].args[0][0], str(Path(env["CUDA_HOME"]) / "bin/nvcc"))
            scoped = native._build_environment(env, toolchain)
            self.assertIn("targets/x86_64-linux/include", scoped["CPATH"])
            self.assertTrue(scoped["CPATH"].endswith("/existing/include"))
            self.assertIn("targets/x86_64-linux/lib", scoped["LIBRARY_PATH"])
            self.assertEqual(scoped["CAUSAL_CONV1D_FORCE_BUILD"], "TRUE")
            self.assertEqual(scoped["CAUSAL_CONV1D_SKIP_CUDA_BUILD"], "FALSE")
            self.assertEqual(scoped["CAUSAL_CONV1D_FORCE_CXX11_ABI"], "TRUE")
            self.assertEqual(scoped["MAX_JOBS"], "2")
            self.assertEqual(env["CPATH"], "/existing/include")

    def test_inherited_nvcc_flags_and_local_version_are_removed_without_mutating_parent_env(self):
        with tempfile.TemporaryDirectory() as folder:
            _, _, _, _, env = fixture(folder)
            overrides = {"NVCC_PREPEND_FLAGS": "-std=c++11", "NVCC_APPEND_FLAGS": "-bad-flag",
                         "CAUSAL_CONV1D_LOCAL_VERSION": "unrelated"}
            env.update(overrides)
            scoped = native._build_environment(env, toolchain_fixture(env))
            for name, value in overrides.items():
                self.assertNotIn(name, scoped)
                self.assertEqual(env[name], value)

    def test_default_and_explicit_compiler_paths_produce_same_toolchain_and_build_fingerprint_inputs(self):
        with tempfile.TemporaryDirectory() as folder:
            _, repo, _, info, env = fixture(folder)
            defaults = {name: value for name, value in env.items() if name not in {"CC", "CXX"}}
            explicit = {**defaults, "CC": shutil.which("gcc", path=env["PATH"]),
                        "CXX": shutil.which("g++", path=env["PATH"])}
            def version(command, repo, env, **kwargs):
                text = "release 12.8, V12.8.61" if command[0].endswith("nvcc") else "compiler version fixture"
                return subprocess.CompletedProcess(command, 0, text, "")
            with patch("lab_native_kernels._run", side_effect=version):
                default_tools = native._toolchain(info, repo, defaults)
                explicit_tools = native._toolchain(info, repo, explicit)
                extra_tools = native._toolchain(info, repo, {**explicit, "CC": explicit["CC"] + " -fno-lto"})
            self.assertEqual(default_tools, explicit_tools)
            default_build = native._build_environment(defaults, default_tools)
            explicit_build = native._build_environment(explicit, explicit_tools)
            self.assertEqual(default_build, explicit_build)
            extra_build = native._build_environment(explicit, extra_tools)
            self.assertEqual(shlex.split(extra_build["CC"]), extra_tools["cc"]["command"][:-1])
            self.assertIn("-fno-lto", shlex.split(extra_build["CC"]))
            self.assertEqual(shlex.split(default_build["CXX"]), default_tools["cxx"]["command"][:-1])

    def test_source_checksum_failure_never_publishes_download(self):
        with tempfile.TemporaryDirectory() as folder:
            cache = Path(folder)
            with patch("lab_native_kernels.urlopen", return_value=io.BytesIO(b"incorrect source bytes")):
                with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                    native._verified_source(cache)
            self.assertEqual(list(cache.iterdir()), [])

    def test_extraction_preserves_license_and_rejects_parent_escape(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for member_name in ("causal_conv1d-1.7.0/LICENSE", "causal_conv1d-1.7.0/../../escape"):
                archive = root / "source.tar.gz"
                with tarfile.open(archive, "w:gz") as target:
                    data = b"third-party license fixture"
                    member = tarfile.TarInfo(member_name)
                    member.size = len(data)
                    target.addfile(member, io.BytesIO(data))
                if ".." in member_name:
                    with self.assertRaisesRegex(ValueError, "Unsafe path"):
                        native._extract_source(archive, root / "unsafe")
                else:
                    extracted = native._extract_source(archive, root / "safe")
                    self.assertEqual((extracted / "LICENSE").read_bytes(), data)

    def test_cached_wheel_requires_matching_fingerprint_and_actual_sha(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            wheel = directory / "causal_conv1d-1.7.0-fixture.whl"
            wheel.write_bytes(b"CPU test artifact, never installed")
            fingerprint = {"source_sha256": native.SOURCE_SHA256, "torch": "2.8.0"}
            receipt = {"fingerprint": fingerprint, "verified_import": True,
                       "wheel": wheel.name, "wheel_sha256": native.file_sha256(wheel)}
            (directory / "receipt.json").write_text(json.dumps(receipt))
            self.assertEqual(native._cache_wheel(directory, fingerprint), wheel)
            with self.assertRaisesRegex(ValueError, "does not match"):
                native._cache_wheel(directory, {**fingerprint, "torch": "changed"})
            wheel.write_bytes(b"modified artifact")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                native._cache_wheel(directory, fingerprint)

    def test_healthy_import_does_not_install_or_build_anything(self):
        with tempfile.TemporaryDirectory() as folder:
            python, repo, workspace, info, env = fixture(folder)
            with patch("lab_native_kernels._run", return_value=subprocess.CompletedProcess([], 0, probe_output(info, True), "")) as run:
                result = native.ensure_native_kernels(python, repo, workspace, env)
            self.assertEqual(result["status"], "already_verified")
            run.assert_called_once()
            self.assertFalse(workspace.exists())

    def test_zero_exit_without_verified_registry_bindings_is_not_admitted(self):
        with tempfile.TemporaryDirectory() as folder:
            python, repo, workspace, info, env = fixture(folder)
            with patch("lab_native_kernels._run", return_value=subprocess.CompletedProcess([], 0, probe_output(info), "")):
                with self.assertRaisesRegex(RuntimeError, "verified optimized"):
                    native.ensure_native_kernels(python, repo, workspace, env)
            self.assertFalse(workspace.exists())

    def test_non_glibc_error_does_not_replace_package_or_hide_original_diagnostic(self):
        with tempfile.TemporaryDirectory() as folder:
            python, repo, workspace, info, env = fixture(folder)
            failed = subprocess.CompletedProcess([], 1, probe_output(info), "undefined symbol: exact original diagnostic")
            with patch("lab_native_kernels._run", return_value=failed) as run, patch("lab_native_kernels.print"):
                with self.assertRaisesRegex(RuntimeError, "exact original diagnostic"):
                    native.ensure_native_kernels(python, repo, workspace, env)
            run.assert_called_once()
            self.assertFalse(workspace.exists())

    def test_only_missing_build_tool_is_installed_without_changing_locked_existing_versions(self):
        with tempfile.TemporaryDirectory() as folder:
            python, repo, workspace, info, env = fixture(folder)
            info["build_tools"] = {"setuptools": "80.9.0", "wheel": None, "packaging": "26.0"}
            ready = {**info, "build_tools": {**info["build_tools"], "wheel": "0.45.1"}}
            failed = subprocess.CompletedProcess([], 1, probe_output(info), "version `GLIBC_2.32' not found")
            verified = subprocess.CompletedProcess([], 0, probe_output(ready, True), "")
            def cached(directory, fingerprint):
                wheel = directory / "causal_conv1d-1.7.0-fixture.whl"
                wheel.write_bytes(b"CPU cache fixture, never executed")
                return wheel
            def run(command, repo, env, **kwargs):
                output = probe_output(ready) if command[0] == python else ""
                return subprocess.CompletedProcess(command, 0, output, "")
            with patch("lab_native_kernels._probe", side_effect=[(failed, info), (verified, ready)]), \
                    patch("lab_native_kernels._toolchain", return_value=toolchain_fixture(env)), \
                    patch("lab_native_kernels._cache_wheel", side_effect=cached), \
                    patch("lab_native_kernels._run", side_effect=run) as calls, patch("lab_native_kernels.print"):
                result = native.ensure_native_kernels(python, repo, workspace, env)
            tool_install = calls.call_args_list[0]
            self.assertEqual(tool_install.args[0], ["uv", "pip", "install", "--python", python, "--no-deps", "wheel==0.45.1"])
            receipt = json.loads(Path(result["receipt"]).read_text())
            self.assertEqual(receipt["fingerprint"]["platform"]["build_tools"]["packaging"], "26.0")
            self.assertEqual(result["status"], "cached_and_verified")

    def test_mocked_build_and_cached_repair_keep_torch_lock_and_all_child_env_bindings(self):
        with tempfile.TemporaryDirectory() as folder:
            python, repo, workspace, info, env = fixture(folder)
            source = Path(folder) / "source"
            source.mkdir()
            imports = 0
            def run(command, cwd, env, **kwargs):
                nonlocal imports
                self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "1")
                self.assertEqual(env["UV_CACHE_DIR"], str(Path(folder) / "uv-cache"))
                if command[0] == str(python) and "-c" in command:
                    imports += 1
                    repaired = imports % 2 == 0
                    return subprocess.CompletedProcess(command, 0 if repaired else 1, probe_output(info, repaired),
                                                       "" if repaired else "version `GLIBC_2.32' not found")
                if command[-1] == "--version":
                    version = "release 12.8, V12.8.61" if command[0].endswith("nvcc") else "gcc 9.4.0"
                    return subprocess.CompletedProcess(command, 0, version, "")
                self.assertEqual(env["MAX_JOBS"], "2")
                self.assertEqual(env["CAUSAL_CONV1D_FORCE_BUILD"], "TRUE")
                if command[:4] == [str(python), "-u", "setup.py", "bdist_wheel"]:
                    self.assertEqual(cwd, source)
                    self.assertEqual(env["VIRTUAL_ENV"], str(python.parent.parent))
                    self.assertEqual(env["CAUSAL_CONV1D_FORCE_CXX11_ABI"], "TRUE")
                    dist = Path(command[command.index("--dist-dir") + 1])
                    dist.mkdir()
                    (dist / "causal_conv1d-1.7.0-fixture.whl").write_bytes(b"CPU mock artifact, never executed")
                else:
                    self.assertEqual(command[:3], ["uv", "pip", "install"])
                    self.assertIn("--no-deps", command)
                    self.assertIn("--reinstall", command)
                    self.assertTrue(command[-1].endswith(".whl"))
                return subprocess.CompletedProcess(command, 0, "", "")
            with patch("lab_native_kernels.subprocess.run", side_effect=run) as calls, \
                    patch("lab_native_kernels._verified_source", return_value=Path(folder) / "verified.tar.gz") as download, \
                    patch("lab_native_kernels._extract_source", return_value=source), patch("lab_native_kernels.print"):
                first = native.ensure_native_kernels(python, repo, workspace, env)
                second = native.ensure_native_kernels(python, repo, workspace, env)
            self.assertEqual(first["status"], "built_and_verified")
            self.assertEqual(second["status"], "cached_and_verified")
            self.assertEqual(sum(call.args[0][:4] == [str(python), "-u", "setup.py", "bdist_wheel"]
                                 for call in calls.call_args_list), 1)
            download.assert_called_once()
            self.assertFalse(any("sync" in call.args[0] for call in calls.call_args_list))
            receipt = json.loads(Path(first["receipt"]).read_text())
            self.assertEqual(receipt["fingerprint"]["source_sha256"], native.SOURCE_SHA256)
            self.assertEqual(receipt["wheel_sha256"], native.file_sha256(first["wheel"]))
            self.assertTrue(receipt["verified_import"])


if __name__ == "__main__":
    unittest.main()
