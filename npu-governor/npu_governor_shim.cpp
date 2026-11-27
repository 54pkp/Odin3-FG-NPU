// SPDX-License-Identifier: GPL-3.0-or-later
// Odin3 NPU Governor: libQnnHtp.so interposer for Snapdragon 8 Elite (SM8750, Hexagon HTP v79).
// Installed under the name libQnnHtp.so; the real Qualcomm library is loaded by absolute path
// (ODIN3_REAL_QNN_HTP / libQnnHtp.real.so next to this file / /usr/lib64/vela-npu/libQnnHtp.so).
// It wraps QnnInterface_getProviders, replaces deviceCreate/deviceFree/deviceGetInfrastructure/graphExecute
// and re-targets the client's HTP DCVS_V3 power vote to the corner chosen in config.bin (seqlock mmap, hot reload).
#include <dlfcn.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#include "HTP/QnnHtpDevice.h"

extern "C" {
__attribute__((used, visibility("default"))) extern const char odin3_npu_governor_marker[];
const char odin3_npu_governor_marker[] = "ODIN3_NPU_GOVERNOR_SHIM_V1";
}

namespace {

constexpr uint32_t kCfgMagic = 0x4F4E5055;
constexpr uint32_t kTelMagic = 0x4F4E544D;

struct Odin3NpuGovConfig {
    uint32_t magic;
    uint32_t version;
    uint32_t seq;
    uint32_t mode;  // 0 passthrough, 1 fixed, 2 smart auto
    uint32_t preset_id;
    uint32_t dcvs_enable;
    uint32_t power_mode;
    uint32_t sleep_disable;
    uint32_t sleep_latency_us;
    uint32_t rpc_latency_us;
    uint32_t hmx_timeout_us;
    uint32_t core_min;
    uint32_t core_target;
    uint32_t core_max;
    uint32_t bus_min;
    uint32_t bus_target;
    uint32_t bus_max;
    uint32_t auto_target_pct;
    uint32_t reserved[14];
};
static_assert(sizeof(Odin3NpuGovConfig) == 128, "config size");

struct Odin3NpuTelemetry {
    uint32_t magic;
    uint32_t version;
    uint64_t timestamp_ns;
    uint32_t pid;
    uint32_t active_tier_idx;
    uint32_t applied_core_target;
    uint32_t applied_core_max;
    uint32_t applied_bus_target;
    uint32_t applied_sleep_disable;
    uint32_t client_idle;
    float infer_fps;
    float avg_exec_ms;
    float p95_exec_ms;
    float duty_cycle_pct;
    uint64_t total_inferences;
    char client_name[32];
    uint32_t pad[6];
};
static_assert(sizeof(Odin3NpuTelemetry) == 128, "telemetry size");

using GetProvidersFn = uint64_t (*)(const void*** list, uint32_t* count);
using DeviceCreateFn = uint64_t (*)(void* log, const void* cfgs, void** dev);
using DeviceFreeFn = uint64_t (*)(void* dev);
using DeviceGetInfraFn = uint64_t (*)(void** infra);
using GraphExecFn = uint64_t (*)(void*, const void*, uint32_t, void*, uint32_t, void*, void*);
using GraphExecAsyncFn = uint64_t (*)(void*, const void*, uint32_t, void*, uint32_t, void*, void*, void*, void*);

constexpr size_t kSlotHeader = 40;
constexpr size_t kSlotDeviceGetInfra = 39, kSlotDeviceCreate = 40, kSlotDeviceFree = 43;
constexpr size_t kSlotGraphExecute = 21, kSlotGraphExecuteAsync = 22;
constexpr size_t kCloneBytes = 40 + 76 * 8;

constexpr uint32_t kCornerDisable = 0x10, kSvs2 = 0x30, kSvs = 0x40, kSvsPlus = 0x50, kNom = 0x60, kNomPlus = 0x70,
                   kTurbo = 0x80, kTurboPlus = 0x90, kMaxCorner = 0xA0;
constexpr uint32_t kPmAdjust = 0x01, kPmSaver = 0x04, kPmPerf = 0x10;

struct AutoLevel {
    uint32_t core_min, core_tgt, bus_min, bus_tgt, power_mode, hmx_us, rpc_us, sleep_us;
};
constexpr AutoLevel kAuto[5] = {
    {kSvs, kNom, kSvs, kSvsPlus, kPmSaver, 220, 450, 650},
    {kSvsPlus, kNomPlus, kSvs, kNom, kPmAdjust, 260, 350, 500},
    {kSvs, kTurbo, kSvs, kNomPlus, kPmAdjust, 350, 250, 350},
    {kSvs, kTurboPlus, kSvs, kTurbo, kPmPerf, 500, 120, 180},
    {kSvs, kMaxCorner, kSvs, kTurboPlus, kPmPerf, 800, 100, 120},
};

uint64_t mono_ns() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return uint64_t(ts.tv_sec) * 1000000000ull + uint64_t(ts.tv_nsec);
}

struct Mapped {
    const Odin3NpuGovConfig* p = nullptr;
};

struct Governor {
    std::mutex mu;
    void* real = nullptr;
    GetProvidersFn real_get_providers = nullptr;
    DeviceCreateFn real_device_create = nullptr;
    DeviceFreeFn real_device_free = nullptr;
    DeviceGetInfraFn real_get_infra = nullptr;
    GraphExecFn real_graph_exec = nullptr;
    GraphExecAsyncFn real_graph_exec_async = nullptr;

    QnnHtpDevice_Infrastructure_t* real_infra = nullptr;
    QnnHtpDevice_Infrastructure_t wrapped_infra{};

    // vote ownership
    uint32_t vote_id = 0;
    bool own_id = false;
    int client_state = 0;  // 0 none, 1 id created / no vote, 2 voted active, 3 released
    std::vector<QnnHtpPerfInfrastructure_PowerConfig_t> client_cfg;

    // config
    Mapped maps[2];
    uint64_t last_map_try_ns = 0;
    Odin3NpuGovConfig cfg{};
    uint32_t applied_seq = 0xFFFFFFFFu;
    bool have_cfg = false;
    bool dirty = true;

    // applied state (telemetry)
    uint32_t applied_core_tgt = 0, applied_core_max = 0, applied_bus_tgt = 0, applied_sleep_dis = 0;
    uint32_t tier_idx = 0;
    bool applied_governed = false;

    // auto scaler
    int level = 3;
    uint32_t win_n = 0;
    double win_exec_us = 0;
    uint64_t win_start_ns = 0;
    int low_windows = 0;
    uint64_t last_step_ns = 0;
    double ema_period_us = 0;
    uint64_t last_call_ns = 0;
    void* primary_graph = nullptr;
    uint64_t last_primary_t0 = 0;
    double cur_pair_exec_us = 0;
    uint32_t pair_win_n = 0;
    bool pair_open = false;

    // telemetry
    uint64_t total = 0;
    float fps = 0, avg_ms = 0, p95_ms = 0, duty = 0;
    double ring[128] = {};
    uint32_t ring_n = 0;
    uint64_t last_flush_ns = 0;
    int tel_fd = -1;
    std::string tel_path;
    char name[32] = {};

    ~Governor() {
        if (tel_fd >= 0) close(tel_fd);
        if (!tel_path.empty()) unlink(tel_path.c_str());
    }
};

Governor& G() {
    static Governor* g = new Governor();  // intentionally leaked: no teardown races with DSP threads
    return *g;
}

// NOTE: no atexit() here: Zig's glibc shim registers it without a DSO handle, so the handler would run from unmapped
// code after the client dlclose()s us (segfault at exit). A destructor attribute runs at dlclose and at process exit.
__attribute__((destructor)) void governor_fini() {
    Governor& g = G();
    if (g.tel_fd >= 0) close(g.tel_fd);
    g.tel_fd = -1;
    if (!g.tel_path.empty()) unlink(g.tel_path.c_str());
    g.tel_path.clear();
}

std::string home_dir() {
    const char* h = getenv("HOME");
    if (h && h[0] == '/') return h;
    return "/var/home/armada";
}

bool same_file(const char* a, const char* b) {
    struct stat sa{}, sb{};
    if (stat(a, &sa) || stat(b, &sb)) return false;
    return sa.st_dev == sb.st_dev && sa.st_ino == sb.st_ino;
}

bool load_real() {
    Governor& g = G();
    if (g.real) return true;
    Dl_info info{};
    dladdr(reinterpret_cast<void*>(&load_real), &info);
    std::string self_path = info.dli_fname ? info.dli_fname : "";
    std::string self_dir = ".";
    auto slash = self_path.find_last_of('/');
    if (slash != std::string::npos) self_dir = self_path.substr(0, slash);

    std::vector<std::string> cands;
    if (const char* env = getenv("ODIN3_REAL_QNN_HTP"); env && env[0] == '/') cands.push_back(env);
    cands.push_back(self_dir + "/libQnnHtp.real.so");
    cands.push_back(home_dir() + "/.local/share/odin3-fg/qairt/lib/libQnnHtp.real.so");
    cands.push_back("/usr/lib64/vela-npu/libQnnHtp.so");
    for (const auto& c : cands) {
        if (access(c.c_str(), R_OK) != 0) continue;
        if (!self_path.empty() && same_file(c.c_str(), self_path.c_str())) continue;
        void* h = dlopen(c.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!h) continue;
        auto fn = reinterpret_cast<GetProvidersFn>(dlsym(h, "QnnInterface_getProviders"));
        if (!fn) {
            dlclose(h);
            continue;
        }
        g.real = h;
        g.real_get_providers = fn;
        return true;
    }
    std::fprintf(stderr, "odin3-npu-governor: real libQnnHtp.so not found\n");
    return false;
}

// ---------- config mapping & seqlock read ----------
void try_map(Governor& g, bool force) {
    uint64_t now = mono_ns();
    if (!force && now - g.last_map_try_ns < 1000000000ull) return;
    g.last_map_try_ns = now;
    const std::string paths[2] = {
        getenv("ODIN3_NPU_GOV_RUN") ? std::string(getenv("ODIN3_NPU_GOV_RUN")) + "/config.bin" : "/run/odin3-npu-governor/config.bin",
        home_dir() + "/.config/odin3-npu-governor/config.bin",
    };
    for (int i = 0; i < 2; ++i) {
        if (g.maps[i].p) continue;
        int fd = open(paths[i].c_str(), O_RDONLY | O_CLOEXEC);
        if (fd < 0) continue;
        struct stat st{};
        if (fstat(fd, &st) == 0 && st.st_size >= 128) {
            void* m = mmap(nullptr, 4096, PROT_READ, MAP_SHARED, fd, 0);
            if (m != MAP_FAILED) g.maps[i].p = static_cast<const Odin3NpuGovConfig*>(m);
        }
        close(fd);
    }
}

bool read_seqlock(const Odin3NpuGovConfig* m, Odin3NpuGovConfig& out) {
    for (int tries = 0; tries < 200; ++tries) {
        uint32_t s1 = __atomic_load_n(&m->seq, __ATOMIC_ACQUIRE);
        if (s1 & 1u) continue;
        std::memcpy(&out, m, sizeof(out));
        __atomic_thread_fence(__ATOMIC_ACQUIRE);
        uint32_t s2 = __atomic_load_n(&m->seq, __ATOMIC_RELAXED);
        if (s1 == s2 && out.magic == kCfgMagic) return true;
    }
    return false;
}

// returns true when a newer config was loaded
bool poll_config(Governor& g) {
    if (!g.maps[0].p || !g.maps[1].p) try_map(g, false);
    const Odin3NpuGovConfig* best = nullptr;
    uint32_t best_seq = 0;
    for (auto& m : g.maps) {
        if (!m.p) continue;
        if (__atomic_load_n(&m.p->magic, __ATOMIC_RELAXED) != kCfgMagic) continue;
        uint32_t s = __atomic_load_n(&m.p->seq, __ATOMIC_ACQUIRE);
        if (s & 1u) continue;
        if (!best || int32_t(s - best_seq) > 0) {
            best = m.p;
            best_seq = s;
        }
    }
    if (!best) return false;
    if (g.have_cfg && best_seq == g.applied_seq) return false;
    Odin3NpuGovConfig tmp{};
    if (!read_seqlock(best, tmp)) return false;
    bool mode_changed = !g.have_cfg || tmp.mode != g.cfg.mode;
    g.cfg = tmp;
    g.have_cfg = true;
    g.applied_seq = tmp.seq;
    g.dirty = true;
    if (mode_changed) {
        g.level = 3;
        g.win_n = 0;
        g.win_exec_us = 0;
        g.low_windows = 0;
    }
    return true;
}

// ---------- applying votes ----------
uint64_t real_set_power(Governor& g, uint32_t id, const QnnHtpPerfInfrastructure_PowerConfig_t** cfgs) {
    if (!g.real_infra) return 1;
    return g.real_infra->perfInfra.setPowerConfig(id, cfgs);
}

uint32_t clamp_corner(uint32_t v, uint32_t lo, uint32_t hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

struct Plan {
    uint32_t dcvs_enable, power_mode, sleep_disable, sleep_us, rpc_us, hmx_us;
    uint32_t core_min, core_tgt, core_max, bus_min, bus_tgt, bus_max;
};

Plan make_plan(Governor& g) {
    const Odin3NpuGovConfig& c = g.cfg;
    Plan p{};
    if (c.mode == 2) {
        const AutoLevel& a = kAuto[std::clamp(g.level, 0, 4)];
        uint32_t cmax = clamp_corner(c.core_max ? c.core_max : kMaxCorner, kSvs, kMaxCorner);
        uint32_t bmax = clamp_corner(c.bus_max ? c.bus_max : kTurboPlus, kSvs2, kMaxCorner);
        p.dcvs_enable = 1;
        p.power_mode = a.power_mode;
        p.sleep_disable = 0;
        p.sleep_us = a.sleep_us;
        p.rpc_us = a.rpc_us;
        p.hmx_us = a.hmx_us;
        p.core_min = std::min(a.core_min, cmax);
        p.core_tgt = std::min(a.core_tgt, cmax);
        p.core_max = p.core_tgt;
        p.bus_min = std::min(a.bus_min, bmax);
        p.bus_tgt = std::min(a.bus_tgt, bmax);
        p.bus_max = p.bus_tgt;
        g.tier_idx = uint32_t(std::clamp(g.level, 0, 4));
    } else {
        p.dcvs_enable = c.dcvs_enable ? 1 : 0;
        p.power_mode = c.power_mode ? c.power_mode : kPmPerf;
        p.sleep_disable = c.sleep_disable ? 1 : 0;
        p.sleep_us = c.sleep_latency_us ? c.sleep_latency_us : 100;
        p.rpc_us = c.rpc_latency_us;
        p.hmx_us = c.hmx_timeout_us;
        p.core_min = clamp_corner(c.core_min, kCornerDisable, kMaxCorner);
        p.core_tgt = clamp_corner(c.core_target, kCornerDisable, kMaxCorner);
        p.core_max = clamp_corner(c.core_max, kCornerDisable, kMaxCorner);
        p.bus_min = clamp_corner(c.bus_min, kCornerDisable, kMaxCorner);
        p.bus_tgt = clamp_corner(c.bus_target, kCornerDisable, kMaxCorner);
        p.bus_max = clamp_corner(c.bus_max, kCornerDisable, kMaxCorner);
        g.tier_idx = c.preset_id;
    }
    if (const char* uc = getenv("ODIN3_NPU_UNCAPPED_COMPACT");
        uc && (uc[0] == '1' || uc[0] == '2') && c.mode != 0) {
        const bool triple = (uc[0] == '2');
        p.dcvs_enable = 1;
        p.power_mode = kPmSaver;
        p.sleep_disable = 0;
        p.sleep_us = triple ? 700u : 850u;
        p.rpc_us = triple ? 300u : 350u;
        p.hmx_us = 180u;
        p.core_min = kSvs;
        p.core_tgt = kSvs;
        p.core_max = kSvsPlus;
        p.bus_min = kSvs2;
        p.bus_tgt = kSvs;
        p.bus_max = kSvs;
    }
    return p;
}

uint64_t apply_governed(Governor& g, uint32_t id) {
    Plan p = make_plan(g);
    // Explicitly disable host ARM64 CPU FastRPC busy-polling so Oryon cores enter WFI during HTP execution!
    QnnHtpPerfInfrastructure_PowerConfig_t pollCfg{};
    pollCfg.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_POLLING_TIME;
    pollCfg.rpcPollingTimeConfig = 0;
    const QnnHtpPerfInfrastructure_PowerConfig_t* pollList[] = {&pollCfg, nullptr};
    (void)real_set_power(g, id, pollList);

    QnnHtpPerfInfrastructure_PowerConfig_t dcvs{};
    dcvs.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3;
    auto& v = dcvs.dcvsV3Config;
    v.contextId = id;
    v.setDcvsEnable = 1;
    v.dcvsEnable = p.dcvs_enable;
    v.powerMode = static_cast<QnnHtpPerfInfrastructure_PowerMode_t>(p.power_mode);
    v.setSleepLatency = 1;
    v.sleepLatency = p.sleep_us;
    v.setSleepDisable = 1;
    v.sleepDisable = p.sleep_disable;
    v.setBusParams = 1;
    v.busVoltageCornerMin = static_cast<QnnHtpPerfInfrastructure_VoltageCorner_t>(p.bus_min);
    v.busVoltageCornerTarget = static_cast<QnnHtpPerfInfrastructure_VoltageCorner_t>(p.bus_tgt);
    v.busVoltageCornerMax = static_cast<QnnHtpPerfInfrastructure_VoltageCorner_t>(p.bus_max);
    v.setCoreParams = 1;
    v.coreVoltageCornerMin = static_cast<QnnHtpPerfInfrastructure_VoltageCorner_t>(p.core_min);
    v.coreVoltageCornerTarget = static_cast<QnnHtpPerfInfrastructure_VoltageCorner_t>(p.core_tgt);
    v.coreVoltageCornerMax = static_cast<QnnHtpPerfInfrastructure_VoltageCorner_t>(p.core_max);

    QnnHtpPerfInfrastructure_PowerConfig_t rpc{};
    rpc.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_CONTROL_LATENCY;
    rpc.rpcControlLatencyConfig = p.rpc_us;
    QnnHtpPerfInfrastructure_PowerConfig_t hmx{};
    hmx.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_HMX_TIMEOUT_INTERVAL_US;
    hmx.hmxTimeoutIntervalUsConfig = p.hmx_us;

    const QnnHtpPerfInfrastructure_PowerConfig_t* list[4];
    int n = 0;
    list[n++] = &dcvs;
    if (p.rpc_us) list[n++] = &rpc;
    if (p.hmx_us) list[n++] = &hmx;
    list[n] = nullptr;
    uint64_t rc = real_set_power(g, id, list);
    if (rc == 0) {
        g.applied_core_tgt = p.core_tgt;
        g.applied_core_max = p.core_max;
        g.applied_bus_tgt = p.bus_tgt;
        g.applied_sleep_dis = p.sleep_disable;
        g.applied_governed = true;
    } else {
        std::fprintf(stderr, "odin3-npu-governor: setPowerConfig rc=0x%llx\n", (unsigned long long)rc);
    }
    return rc;
}

uint64_t apply_passthrough(Governor& g, uint32_t id) {
    if (!g.client_cfg.empty()) {
        std::vector<const QnnHtpPerfInfrastructure_PowerConfig_t*> list;
        for (auto& c : g.client_cfg) {
            list.push_back(&c);
            if (c.option == QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3) const_cast<QnnHtpPerfInfrastructure_PowerConfig_t&>(c).dcvsV3Config.contextId = id;
        }
        list.push_back(nullptr);
        uint64_t rc = real_set_power(g, id, list.data());
        if (rc == 0) {
            for (auto& c : g.client_cfg) {
                if (c.option == QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3) {
                    g.applied_core_tgt = c.dcvsV3Config.coreVoltageCornerTarget;
                    g.applied_core_max = c.dcvsV3Config.coreVoltageCornerMax;
                    g.applied_bus_tgt = c.dcvsV3Config.busVoltageCornerTarget;
                    g.applied_sleep_dis = c.dcvsV3Config.sleepDisable;
                }
            }
        }
        g.applied_governed = false;
        return rc;
    }
    // we created the vote ourselves and are now switched off: release it like the Vela daemon does
    QnnHtpPerfInfrastructure_PowerConfig_t rel{};
    rel.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3;
    auto& v = rel.dcvsV3Config;
    v.contextId = id;
    v.setDcvsEnable = 1;
    v.dcvsEnable = 1;
    v.powerMode = QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_POWER_SAVER_MODE;
    v.setSleepLatency = 1;
    v.sleepLatency = 1000;
    v.setSleepDisable = 1;
    v.sleepDisable = 0;
    v.setBusParams = 1;
    v.busVoltageCornerMin = v.busVoltageCornerTarget = v.busVoltageCornerMax = DCVS_VOLTAGE_CORNER_DISABLE;
    v.setCoreParams = 1;
    v.coreVoltageCornerMin = v.coreVoltageCornerTarget = v.coreVoltageCornerMax = DCVS_VOLTAGE_CORNER_DISABLE;
    const QnnHtpPerfInfrastructure_PowerConfig_t* list[] = {&rel, nullptr};
    g.applied_governed = false;
    g.applied_core_tgt = g.applied_core_max = g.applied_bus_tgt = kCornerDisable;
    g.applied_sleep_dis = 0;
    return real_set_power(g, id, list);
}

void ensure_real_infra(Governor& g) {
    if (g.real_infra || !g.real_get_infra) return;
    void* inf = nullptr;
    if (g.real_get_infra(&inf) == 0 && inf) {
        auto* ri = static_cast<QnnHtpDevice_Infrastructure_t*>(inf);
        if (ri->infraType == QNN_HTP_DEVICE_INFRASTRUCTURE_TYPE_PERF) g.real_infra = ri;
    }
}

// called with g.mu held
void apply_now(Governor& g) {
    if (!g.have_cfg) return;
    ensure_real_infra(g);
    if (!g.real_infra) return;
    if (g.vote_id == 0 && g.client_state == 0 && g.cfg.mode != 0) {
        uint32_t id = 0;
        if (g.real_infra->perfInfra.createPowerConfigId(0, 0, &id) == 0) {
            g.vote_id = id;
            g.own_id = true;
            g.client_state = 2;
        }
    }
    if (g.vote_id == 0 || g.client_state != 2) {
        g.dirty = false;
        return;
    }
    if (g.cfg.mode == 0) {
        if (g.applied_governed || g.own_id) apply_passthrough(g, g.vote_id);
    } else {
        apply_governed(g, g.vote_id);
    }
    g.dirty = false;
}

// ---------- telemetry ----------
void flush_telemetry(Governor& g, uint64_t now) {
    Odin3NpuTelemetry t{};
    t.magic = kTelMagic;
    t.version = 1;
    t.timestamp_ns = now;
    t.pid = uint32_t(getpid());
    t.active_tier_idx = g.tier_idx;
    t.applied_core_target = g.applied_core_tgt;
    t.applied_core_max = g.applied_core_max;
    t.applied_bus_target = g.applied_bus_tgt;
    t.applied_sleep_disable = g.applied_sleep_dis;
    t.client_idle = (g.client_state == 3 || g.client_state == 1) ? 1 : 0;
    t.infer_fps = g.fps;
    t.avg_exec_ms = g.avg_ms;
    t.p95_exec_ms = g.p95_ms;
    t.duty_cycle_pct = g.duty;
    t.total_inferences = g.total;
    std::memcpy(t.client_name, g.name, sizeof(t.client_name));
    if (g.tel_fd < 0) {
        const char* run = getenv("ODIN3_NPU_GOV_RUN");
        std::string pid = std::to_string(getpid());
        std::string cands[2] = {
            std::string(run ? run : "/run/odin3-npu-governor") + "/telemetry/pid-" + pid + ".bin",
            home_dir() + "/.cache/odin3-npu-governor/telemetry/pid-" + pid + ".bin",
        };
        for (auto& c : cands) {
            int fd = open(c.c_str(), O_WRONLY | O_CREAT | O_CLOEXEC, 0644);
            if (fd >= 0) {
                g.tel_fd = fd;
                g.tel_path = c;
                break;
            }
        }
    }
    if (g.tel_fd >= 0) {
        ssize_t r = pwrite(g.tel_fd, &t, sizeof(t), 0);
        (void)r;
    }
}

void init_name(Governor& g) {
    if (g.name[0]) return;
    const char* env = getenv("ODIN3_NPU_CLIENT_NAME");
    if (env && env[0]) {
        std::snprintf(g.name, sizeof(g.name), "%s", env);
        return;
    }
    FILE* f = fopen("/proc/self/comm", "r");
    if (f) {
        if (fgets(g.name, sizeof(g.name), f)) {
            size_t n = std::strlen(g.name);
            while (n && (g.name[n - 1] == '\n' || g.name[n - 1] == ' ')) g.name[--n] = 0;
        }
        fclose(f);
    }
}

// ---------- hooks ----------
uint64_t hook_device_create(void* log, const void* cfgs, void** dev) {
    Governor& g = G();
    uint64_t rc = g.real_device_create(log, cfgs, dev);
    return rc;
}

uint64_t hook_device_free(void* dev) {
    Governor& g = G();
    {
        std::lock_guard<std::mutex> lk(g.mu);
        if (g.own_id && g.vote_id && g.real_infra) {
            g.real_infra->perfInfra.destroyPowerConfigId(g.vote_id);
            g.vote_id = 0;
            g.own_id = false;
            g.client_state = 0;
        }
        g.primary_graph = nullptr;
        g.pair_open = false;
        g.last_primary_t0 = 0;
        g.dirty = true;
    }
    return g.real_device_free(dev);
}

uint64_t w_create_power_config_id(uint32_t deviceId, uint32_t coreId, uint32_t* id) {
    Governor& g = G();
    uint64_t rc = g.real_infra->perfInfra.createPowerConfigId(deviceId, coreId, id);
    if (rc == 0 && id) {
        std::lock_guard<std::mutex> lk(g.mu);
        if (g.own_id && g.vote_id) {
            g.real_infra->perfInfra.destroyPowerConfigId(g.vote_id);
            g.own_id = false;
        }
        g.vote_id = *id;
        g.client_state = 1;
    }
    return rc;
}

uint64_t w_destroy_power_config_id(uint32_t id) {
    Governor& g = G();
    {
        std::lock_guard<std::mutex> lk(g.mu);
        if (id == g.vote_id) {
            g.vote_id = 0;
            g.client_state = 0;
            g.client_cfg.clear();
            g.applied_governed = false;
        }
    }
    return g.real_infra->perfInfra.destroyPowerConfigId(id);
}

uint64_t w_set_power_config(uint32_t id, const QnnHtpPerfInfrastructure_PowerConfig_t** config) {
    Governor& g = G();
    std::lock_guard<std::mutex> lk(g.mu);
    if (!config) return g.real_infra->perfInfra.setPowerConfig(id, config);
    poll_config(g);
    bool has_dcvs = false, release = false;
    std::vector<QnnHtpPerfInfrastructure_PowerConfig_t> copy;
    for (size_t i = 0; config[i] && i < 8; ++i) {
        copy.push_back(*config[i]);
        if (config[i]->option == QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3) {
            has_dcvs = true;
            const auto& v = config[i]->dcvsV3Config;
            if (v.setCoreParams && v.coreVoltageCornerMax == DCVS_VOLTAGE_CORNER_DISABLE &&
                v.setBusParams && v.busVoltageCornerMax == DCVS_VOLTAGE_CORNER_DISABLE)
                release = true;
        }
    }
    if (!has_dcvs) return g.real_infra->perfInfra.setPowerConfig(id, config);
    g.vote_id = id;
    if (release) {
        g.client_state = 3;
        g.applied_governed = false;
        g.applied_core_tgt = g.applied_core_max = g.applied_bus_tgt = kCornerDisable;
        return g.real_infra->perfInfra.setPowerConfig(id, config);
    }
    g.client_cfg = copy;
    g.client_state = 2;
    if (!g.have_cfg || g.cfg.mode == 0) {
        uint64_t rc = g.real_infra->perfInfra.setPowerConfig(id, config);
        for (auto& c : copy) {
            if (c.option == QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3) {
                g.applied_core_tgt = c.dcvsV3Config.coreVoltageCornerTarget;
                g.applied_core_max = c.dcvsV3Config.coreVoltageCornerMax;
                g.applied_bus_tgt = c.dcvsV3Config.busVoltageCornerTarget;
                g.applied_sleep_dis = c.dcvsV3Config.sleepDisable;
            }
        }
        g.applied_governed = false;
        return rc;
    }
    uint64_t rc = apply_governed(g, id);
    if (rc != 0) rc = g.real_infra->perfInfra.setPowerConfig(id, config);
    g.dirty = false;
    return rc;
}

uint64_t w_set_memory_config(uint32_t d, uint32_t c, const QnnHtpPerfInfrastructure_MemoryConfig_t** cfg) {
    Governor& g = G();
    return g.real_infra->perfInfra.setMemoryConfig(d, c, cfg);
}

uint64_t hook_get_infra(void** infra) {
    Governor& g = G();
    if (!infra) return g.real_get_infra(infra);
    void* real = nullptr;
    uint64_t rc = g.real_get_infra(&real);
    if (rc != 0 || !real) {
        *infra = real;
        return rc;
    }
    std::lock_guard<std::mutex> lk(g.mu);
    auto* ri = static_cast<QnnHtpDevice_Infrastructure_t*>(real);
    if (ri->infraType != QNN_HTP_DEVICE_INFRASTRUCTURE_TYPE_PERF) {
        *infra = real;
        return rc;
    }
    g.real_infra = ri;
    g.wrapped_infra = *ri;
    g.wrapped_infra.perfInfra.createPowerConfigId = &w_create_power_config_id;
    g.wrapped_infra.perfInfra.destroyPowerConfigId = &w_destroy_power_config_id;
    g.wrapped_infra.perfInfra.setPowerConfig = &w_set_power_config;
    g.wrapped_infra.perfInfra.setMemoryConfig = &w_set_memory_config;
    *infra = &g.wrapped_infra;
    return rc;
}

void pre_exec(Governor& g) {
    // cheap lock-free check first
    bool changed = false;
    {
        std::lock_guard<std::mutex> lk(g.mu);
        changed = poll_config(g);
        if (g.dirty || changed) {
            init_name(g);
            apply_now(g);
        }
    }
}

// Pair accounting: a frame-generation "pair" is 1..3 graphExecute calls (joined; mot+p12; mot+p13+p23). The first graph seen
// is the primary graph and marks the start of a pair, so duty/FPS/auto-scaling see the full pair cost against the true
// inter-pair period.
void close_pair(Governor& g, uint64_t t_now) {
    if (!g.pair_open) return;
    g.pair_open = false;
    ++g.total;
    g.ring[g.ring_n++ & 127u] = g.cur_pair_exec_us;
    ++g.pair_win_n;
    const char* uc = getenv("ODIN3_NPU_UNCAPPED_COMPACT");
    bool auto_mode = g.have_cfg && g.cfg.mode == 2 && g.client_state == 2 && !(uc && (uc[0] == '1' || uc[0] == '2'));
    if (auto_mode && g.level < 4 && g.ema_period_us > 0 && g.ema_period_us < 40000.0 &&
        g.cur_pair_exec_us > 0.72 * g.ema_period_us && t_now - g.last_step_ns > 100000000ull) {
        ++g.level;
        g.last_step_ns = t_now;
        g.low_windows = 0;
        apply_now(g);
    }
    if (g.pair_win_n >= 20) {
        double wall_us = double(t_now - g.win_start_ns) / 1000.0;
        if (wall_us > 0) {
            double duty = std::min(100.0, 100.0 * g.win_exec_us / wall_us);
            g.duty = float(duty);
            g.fps = float(g.pair_win_n * 1e6 / wall_us);
            g.avg_ms = float(g.win_exec_us / g.pair_win_n / 1000.0);
            uint32_t n = std::min<uint32_t>(g.ring_n, 128u);
            double tmp[128];
            std::memcpy(tmp, g.ring, n * sizeof(double));
            std::sort(tmp, tmp + n);
            g.p95_ms = n ? float(tmp[std::min<uint32_t>(n - 1, uint32_t(n * 0.95))] / 1000.0) : 0.f;
            if (auto_mode && wall_us < 2000000.0) {
                uint32_t target = g.cfg.auto_target_pct ? g.cfg.auto_target_pct : 75;
                const int min_lvl = (g.ema_period_us > 0.0 && g.ema_period_us < 30000.0 && g.avg_ms > 6.0f) ? 1 : 0;
                if (duty > double(target) + 10.0 && g.level < 4) {
                    ++g.level;
                    g.low_windows = 0;
                    g.last_step_ns = t_now;
                    apply_now(g);
                } else if (duty < double(target) - 10.0) {
                    if (++g.low_windows >= 2 && g.level > min_lvl) {
                        --g.level;
                        g.low_windows = 0;
                        g.last_step_ns = t_now;
                        apply_now(g);
                    }
                } else {
                    g.low_windows = 0;
                }
            }
        }
        g.pair_win_n = 0;
        g.win_exec_us = 0;
        g.win_start_ns = t_now;
    }
}

void post_exec(Governor& g, void* graph, uint64_t t0, uint64_t t1) {
    std::lock_guard<std::mutex> lk(g.mu);
    double exec_us = double(t1 - t0) / 1000.0;
    if (!g.primary_graph) g.primary_graph = graph;
    if (graph == g.primary_graph) {
        close_pair(g, t0);
        if (g.last_primary_t0) {
            double period = double(t0 - g.last_primary_t0) / 1000.0;
            if (period < 500000.0) g.ema_period_us = g.ema_period_us == 0 ? period : g.ema_period_us * 0.9 + period * 0.1;
            else g.win_start_ns = t0;  // long idle: restart the duty window
        }
        if (g.pair_win_n == 0 && g.win_start_ns == 0) g.win_start_ns = t0;
        g.last_primary_t0 = t0;
        g.pair_open = true;
        g.cur_pair_exec_us = 0;
    }
    g.cur_pair_exec_us += exec_us;
    g.win_exec_us += exec_us;
    if (t1 - g.last_flush_ns > 450000000ull) {
        g.last_flush_ns = t1;
        flush_telemetry(g, t1);
    }
}
uint64_t hook_graph_exec(void* graph, const void* in, uint32_t nin, void* out, uint32_t nout, void* prof, void* sig) {
    Governor& g = G();
    pre_exec(g);
    uint64_t t0 = mono_ns();
    uint64_t rc = g.real_graph_exec(graph, in, nin, out, nout, prof, sig);
    post_exec(g, graph, t0, mono_ns());
    return rc;
}

uint64_t hook_graph_exec_async(void* graph, const void* in, uint32_t nin, void* out, uint32_t nout, void* prof, void* sig,
                               void* notify, void* param) {
    Governor& g = G();
    pre_exec(g);
    return g.real_graph_exec_async(graph, in, nin, out, nout, prof, sig, notify, param);
}

}  // namespace

extern "C" __attribute__((visibility("default"))) uint64_t QnnInterface_getProviders(const void*** list, uint32_t* count) {
    static std::once_flag once;
    static uint8_t clone[1024];
    static const void* clone_list[2];
    static uint64_t cached_rc = 1;
    static uint32_t cached_count = 0;
    Governor& g = G();
    std::call_once(once, [&] {
        if (!load_real()) return;
        const void** real_list = nullptr;
        uint32_t n = 0;
        cached_rc = g.real_get_providers(&real_list, &n);
        if (cached_rc != 0 || !n || !real_list) return;
        std::memset(clone, 0, sizeof(clone));
        std::memcpy(clone, real_list[0], kCloneBytes);
        auto** tbl = reinterpret_cast<void**>(clone + kSlotHeader);
        g.real_device_create = reinterpret_cast<DeviceCreateFn>(tbl[kSlotDeviceCreate]);
        g.real_device_free = reinterpret_cast<DeviceFreeFn>(tbl[kSlotDeviceFree]);
        g.real_get_infra = reinterpret_cast<DeviceGetInfraFn>(tbl[kSlotDeviceGetInfra]);
        g.real_graph_exec = reinterpret_cast<GraphExecFn>(tbl[kSlotGraphExecute]);
        g.real_graph_exec_async = reinterpret_cast<GraphExecAsyncFn>(tbl[kSlotGraphExecuteAsync]);
        if (g.real_device_create) tbl[kSlotDeviceCreate] = reinterpret_cast<void*>(&hook_device_create);
        if (g.real_device_free) tbl[kSlotDeviceFree] = reinterpret_cast<void*>(&hook_device_free);
        if (g.real_get_infra) tbl[kSlotDeviceGetInfra] = reinterpret_cast<void*>(&hook_get_infra);
        if (g.real_graph_exec) tbl[kSlotGraphExecute] = reinterpret_cast<void*>(&hook_graph_exec);
        if (g.real_graph_exec_async) tbl[kSlotGraphExecuteAsync] = reinterpret_cast<void*>(&hook_graph_exec_async);
        clone_list[0] = clone;
        clone_list[1] = nullptr;
        cached_count = 1;
        try_map(g, true);
    });
    if (cached_rc != 0 || !cached_count) return cached_rc ? cached_rc : 1;
    if (list) *list = clone_list;
    if (count) *count = cached_count;
    return 0;
}
