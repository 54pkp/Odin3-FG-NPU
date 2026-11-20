"""Build the pinned Vela daemon with explicit SM8750/v79 device selection."""
import subprocess

from build_inputs import BUILD, LOCAL, SOURCES, VULKAN_HEADERS, ZIG, location


def main():
    source = SOURCES / "vela/npu/src"
    sdk = location("ODIN3_QAIRT_SDK", LOCAL / "qairt-sdk")
    include = location("ODIN3_QNN_INCLUDE", sdk / "include/QNN")
    if not (include / "QnnInterface.h").is_file():
        raise FileNotFoundError(f"Could not find the QAIRT headers at {include}")
    output = BUILD / "daemon"
    output.mkdir(parents=True, exist_ok=True)
    code = (source / "fsr4npud.cpp").read_text(encoding="utf-8")
    needle = "Q.deviceCreate(nullptr, nullptr, &DV);"
    if code.count(needle) != 1:
        raise ValueError("Expected one device-configuration site in the pinned daemon")
    code = code.replace(needle, """QnnHtpDevice_CustomConfig_t soc{};
  soc.option = QNN_HTP_DEVICE_CONFIG_OPTION_SOC;
  soc.socModel = QNN_SOC_MODEL_SM8750;
  QnnHtpDevice_CustomConfig_t pd{};
  pd.option = QNN_HTP_DEVICE_CONFIG_OPTION_SIGNEDPD;
  pd.useSignedProcessDomain.deviceId = 0;
  pd.useSignedProcessDomain.useSignedProcessDomain = false;
  const QnnDevice_Config_t sc{QNN_DEVICE_CONFIG_OPTION_CUSTOM, &soc};
  const QnnDevice_Config_t pc{QNN_DEVICE_CONFIG_OPTION_CUSTOM, &pd};
  const QnnDevice_Config_t* configs[] = { &sc, &pc, nullptr };
  QK(Q.deviceCreate(nullptr, configs, &DV));""")
    code = code.replace("if (bind(ls, (sockaddr*)&a", "if (::bind(ls, (sockaddr*)&a")
    entry = output / "anvil-npud.cpp"
    entry.write_text(code, encoding="utf-8")
    args = [ZIG, "c++", "-target", "aarch64-linux-gnu.2.39", "-O2", "-std=c++17",
            "-I", VULKAN_HEADERS, "-I", include, "-I", source, entry,
            "-o", output / "anvil-npud", "-ldl", "-lpthread", "-lc"]
    subprocess.run([str(item) for item in args], check=True)
    print(output / "anvil-npud")


if __name__ == "__main__":
    main()
