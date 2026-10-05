"""Unit tests for D-4: GGUF tool discovery + command construction (no heavy deps)."""

import sys

import pytest

from generation import quantize_gguf as qg


def test_config_output_paths():
    config = qg.QuantizeConfig()
    assert config.f16_gguf.endswith("finsight-llama-3.2-3b-f16.gguf")
    assert config.quant_gguf.endswith("finsight-llama-3.2-3b-q4_k_m.gguf")


def test_find_binaries_requires_directory():
    with pytest.raises(FileNotFoundError):
        qg.find_llama_cpp_binaries(None)


def test_find_binaries_missing_dir():
    with pytest.raises(FileNotFoundError):
        qg.find_llama_cpp_binaries("C:/definitely/not/here")


def test_find_binaries_success(tmp_path):
    (tmp_path / "convert_hf_to_gguf.py").write_text("# converter", encoding="utf-8")
    (tmp_path / "llama-quantize").write_text("#!/bin/sh", encoding="utf-8")

    tools = qg.find_llama_cpp_binaries(str(tmp_path))
    assert tools["converter"].endswith("convert_hf_to_gguf.py")
    assert tools["quantizer"].endswith("llama-quantize")


def test_find_binaries_missing_quantizer(tmp_path):
    (tmp_path / "convert_hf_to_gguf.py").write_text("# converter", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        qg.find_llama_cpp_binaries(str(tmp_path))


def test_build_conversion_commands(tmp_path):
    config = qg.QuantizeConfig(
        merged_dir=str(tmp_path / "merged"),
        export_dir=str(tmp_path / "gguf"),
        quant_type="Q4_K_M",
    )
    tools = {"converter": "convert.py", "quantizer": "quantize"}
    commands = qg.build_conversion_commands(config, tools)

    assert len(commands) == 2
    convert, quantize = commands
    assert convert[0] == sys.executable
    assert "convert.py" in convert
    assert "--outtype" in convert and "f16" in convert
    assert quantize[0] == "quantize"
    assert quantize[-1] == "Q4_K_M"
    assert (tmp_path / "gguf").exists()
