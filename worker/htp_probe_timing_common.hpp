// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <algorithm>
#include <array>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#include <dlfcn.h>
#include <fcntl.h>
#include <linux/dma-buf.h>
#include <linux/dma-heap.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include "QnnInterface.h"
#include "System/QnnSystemInterface.h"
#include "HTP/QnnHtpCommon.h"
#include "HTP/QnnHtpDevice.h"
#include "HTP/QnnHtpMem.h"

namespace htp_probe {
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
constexpr uint32_t height = 720, width = 1280;

inline void check(Qnn_ErrorHandle_t code, const char* operation) {
    if (code != QNN_SUCCESS) {
        char message[128];
        std::snprintf(message, sizeof(message), "%s: QNN status 0x%llx", operation,
                      static_cast<unsigned long long>(code));
        throw std::runtime_error(message);
    }
}

inline double elapsed_us(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::micro>(end - start).count();
}

inline fs::path qairt_library_path(const char* name) {
    const char* configured = std::getenv("ODIN3_QAIRT_LIB_DIR");
    if (!configured)
        return name;
    const fs::path directory(configured);
    if (!directory.is_absolute())
        throw std::runtime_error("ODIN3_QAIRT_LIB_DIR must be an absolute path");
    return directory / name;
}

class Library {
    void* handle;
public:
    explicit Library(const fs::path& path) : handle(dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL)) {
        if (!handle)
            throw std::runtime_error(dlerror());
    }
    Library(const Library&) = delete;
    Library& operator=(const Library&) = delete;
    ~Library() { dlclose(handle); }
    template<class Function> Function load(const char* name) const {
        auto result = reinterpret_cast<Function>(dlsym(handle, name));
        if (!result)
            throw std::runtime_error(dlerror());
        return result;
    }
};

struct Htp {
    Library htp{qairt_library_path("libQnnHtp.so")};
    Library system{qairt_library_path("libQnnSystem.so")};
    QNN_INTERFACE_VER_TYPE q{};
    QNN_SYSTEM_INTERFACE_VER_TYPE s{};
    Qnn_BackendHandle_t backend{};
    Qnn_DeviceHandle_t device{};
    Qnn_ContextHandle_t context{};
    QnnSystemContext_Handle_t metadata{};
    const QnnHtpDevice_PerfInfrastructure_t* performance{};
    std::optional<uint32_t> power_id;

    ~Htp() {
        if (power_id) {
            const auto status = performance->destroyPowerConfigId(*power_id);
            if (status != QNN_SUCCESS)
                std::fprintf(stderr, "Release HTP power vote: QNN status 0x%llx\n",
                             static_cast<unsigned long long>(status));
        }
        if (context) q.contextFree(context, nullptr);
        if (device) q.deviceFree(device);
        if (backend) q.backendFree(backend);
        if (metadata) s.systemContextFree(metadata);
    }

    void open() {
        const QnnInterface_t** providers = nullptr;
        uint32_t count = 0;
        const auto get_htp = htp.load<decltype(&QnnInterface_getProviders)>("QnnInterface_getProviders");
        check(get_htp(&providers, &count), "Get HTP providers");
        if (!count || !providers)
            throw std::runtime_error("No HTP providers");
        const auto end = providers + count;
        const auto provider = std::find_if(providers, end, [](const QnnInterface_t* item) {
            return item && item->backendId == QNN_BACKEND_ID_HTP &&
                   item->apiVersion.coreApiVersion.major == QNN_API_VERSION_MAJOR &&
                   item->apiVersion.coreApiVersion.minor >= QNN_API_VERSION_MINOR;
        });
        if (provider == end)
            throw std::runtime_error("No compatible QNN HTP provider");
        q = (*provider)->QNN_INTERFACE_VER_NAME;
        const QnnSystemInterface_t** systems = nullptr;
        const auto get_system = system.load<decltype(&QnnSystemInterface_getProviders)>("QnnSystemInterface_getProviders");
        check(get_system(&systems, &count), "Get QNN system providers");
        if (!count || !systems)
            throw std::runtime_error("No QNN system providers");
        const auto system_end = systems + count;
        const auto info = std::find_if(systems, system_end, [](const QnnSystemInterface_t* item) {
            return item && item->systemApiVersion.major == QNN_SYSTEM_API_VERSION_MAJOR &&
                   item->systemApiVersion.minor >= QNN_SYSTEM_API_VERSION_MINOR;
        });
        if (info == system_end)
            throw std::runtime_error("No compatible QNN system provider");
        s = (*info)->QNN_SYSTEM_INTERFACE_VER_NAME;

        check(q.backendCreate(nullptr, nullptr, &backend), "Create HTP backend");
        QnnHtpDevice_CustomConfig_t soc{};
        soc.option = QNN_HTP_DEVICE_CONFIG_OPTION_SOC;
        soc.socModel = QNN_SOC_MODEL_SM8750;
        QnnHtpDevice_CustomConfig_t domain{};
        domain.option = QNN_HTP_DEVICE_CONFIG_OPTION_SIGNEDPD;
        domain.useSignedProcessDomain.deviceId = 0;
        domain.useSignedProcessDomain.useSignedProcessDomain = false;
        const QnnDevice_Config_t sc{QNN_DEVICE_CONFIG_OPTION_CUSTOM, {&soc}};
        const QnnDevice_Config_t dc{QNN_DEVICE_CONFIG_OPTION_CUSTOM, {&domain}};
        const QnnDevice_Config_t* configs[] = {&sc, &dc, nullptr};
        check(q.deviceCreate(nullptr, configs, &device), "Create unsigned SM8750 HTP device");
    }

    void enable_top_power() {
        QnnDevice_Infrastructure_t infrastructure = nullptr;
        if (!q.deviceGetInfrastructure)
            throw std::runtime_error("HTP power infrastructure is unavailable");
        check(q.deviceGetInfrastructure(&infrastructure), "Get HTP power infrastructure");
        if (!infrastructure)
            throw std::runtime_error("HTP power infrastructure is unavailable");
        performance = &reinterpret_cast<const QnnHtpDevice_Infrastructure_t*>(infrastructure)->perfInfra;
        if (!performance->createPowerConfigId || !performance->destroyPowerConfigId || !performance->setPowerConfig)
            throw std::runtime_error("HTP power configuration is unavailable");
        uint32_t id = 0;
        check(performance->createPowerConfigId(0, 0, &id), "Create process-scoped HTP power configuration");
        power_id = id;

        const char* lockMax = std::getenv("ODIN3_HTP_LOCK_MAX_POWER");
        const bool forceLockedMax = lockMax && std::string_view(lockMax) == "1";

        QnnHtpPerfInfrastructure_PowerConfig_t config{};
        config.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3;
        auto& vote = config.dcvsV3Config;
        vote.contextId = id;
        vote.setDcvsEnable = 1;
        vote.dcvsEnable = forceLockedMax ? 0 : 1;
        vote.powerMode = QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE;
        vote.setSleepLatency = 1;
        vote.sleepLatency = forceLockedMax ? 40 : 100;
        vote.setSleepDisable = 1;
        vote.sleepDisable = forceLockedMax ? 1 : 0;
        vote.setBusParams = 1;
        vote.busVoltageCornerMin = forceLockedMax
            ? DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER : DCVS_VOLTAGE_VCORNER_SVS;
        vote.busVoltageCornerTarget = forceLockedMax
            ? DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER : DCVS_VOLTAGE_VCORNER_TURBO_PLUS;
        vote.busVoltageCornerMax = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
        vote.setCoreParams = 1;
        vote.coreVoltageCornerMin = forceLockedMax
            ? DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER : DCVS_VOLTAGE_VCORNER_SVS;
        vote.coreVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
        vote.coreVoltageCornerMax = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;

        QnnHtpPerfInfrastructure_PowerConfig_t rpcConfig{};
        rpcConfig.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_CONTROL_LATENCY;
        rpcConfig.rpcControlLatencyConfig = 100;

        QnnHtpPerfInfrastructure_PowerConfig_t pollConfig{};
        pollConfig.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_POLLING_TIME;
        pollConfig.rpcPollingTimeConfig = 0;
        const QnnHtpPerfInfrastructure_PowerConfig_t* pollConfigs[] = {&pollConfig, nullptr};
        (void)performance->setPowerConfig(id, pollConfigs);

        const QnnHtpPerfInfrastructure_PowerConfig_t* configs[] = {&config, &rpcConfig, nullptr};
        check(performance->setPowerConfig(id, configs), "Apply Gen-5 dynamic HTP power vote");
    }
};

inline std::vector<uint8_t> read_file(const fs::path& path) {
    auto file = std::ifstream(path, std::ios::binary | std::ios::ate);
    const auto size = file.tellg();
    if (size <= 0)
        throw std::runtime_error("Cannot read " + path.string());
    std::vector<uint8_t> data(static_cast<std::size_t>(size));
    if (!file.seekg(0) || !file.read(reinterpret_cast<char*>(data.data()), size))
        throw std::runtime_error("Cannot read " + path.string());
    return data;
}

class DmaArena {
    const Htp& htp;
    const std::size_t bytes;
    int fd = -1;
    void* mapping = MAP_FAILED;
    std::vector<Qnn_MemHandle_t> handles;

public:
    DmaArena(const Htp& backend, std::size_t length, std::size_t registrations = 6,
             bool map_for_cpu = true, int external_fd = -1)
        : htp(backend), bytes(length) {
        handles.reserve(registrations);
        if (external_fd >= 0) {
            fd = fcntl(external_fd, F_DUPFD_CLOEXEC, 0);
            if (fd < 0)
                throw std::system_error(errno, std::generic_category(), "Duplicate HTP DMA-BUF");
        } else {
            const int heap = open("/dev/dma_heap/system", O_RDONLY | O_CLOEXEC);
            if (heap < 0)
                throw std::system_error(errno, std::generic_category(), "Open system DMA heap");
            dma_heap_allocation_data allocation{};
            allocation.len = length;
            allocation.fd_flags = O_RDWR | O_CLOEXEC;
            if (ioctl(heap, DMA_HEAP_IOCTL_ALLOC, &allocation) < 0) {
                const int error = errno;
                close(heap);
                throw std::system_error(error, std::generic_category(), "Allocate system DMA-BUF");
            }
            close(heap);
            fd = static_cast<int>(allocation.fd);
        }
        try {
            if (map_for_cpu)
                map_cpu();
        } catch (...) {
            close(fd);
            throw;
        }
    }
    DmaArena(const DmaArena&) = delete;
    DmaArena& operator=(const DmaArena&) = delete;
    ~DmaArena() {
        for (const auto& handle : handles) {
            if (!handle)
                continue;
            const auto status = htp.q.memDeRegister(&handle, 1);
            if (status != QNN_SUCCESS)
                std::fprintf(stderr, "Deregister HTP DMA-BUF: QNN status 0x%llx\n",
                             static_cast<unsigned long long>(status));
        }
        if (mapping != MAP_FAILED)
            munmap(mapping, bytes);
        close(fd);
    }

    std::size_t size() const { return bytes; }
    int borrow_fd() const { return fd; }
    char* at(std::size_t offset) const { return static_cast<char*>(mapping) + offset; }

    void map_cpu() {
        if (mapping == MAP_FAILED) {
            mapping = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
            if (mapping == MAP_FAILED)
                throw std::system_error(errno, std::generic_category(), "Map system DMA-BUF");
        }
    }

    void sync(__u64 flags) const {
        dma_buf_sync request{flags};
        int status;
        do {
            status = ioctl(fd, DMA_BUF_IOCTL_SYNC, &request);
        } while (status < 0 && (errno == EINTR || errno == EAGAIN));
        if (status < 0)
            throw std::system_error(errno, std::generic_category(), "Synchronize CPU access to system DMA-BUF");
    }

    template<class Descriptor>
    void register_tensor(Descriptor& desc, Qnn_ContextHandle_t context, std::size_t offset,
                         std::size_t tensor_bytes) {
        if (offset >= bytes || tensor_bytes > bytes - offset)
            throw std::runtime_error("Tensor exceeds DMA-BUF capacity");
        QnnMemHtp_Descriptor_t shared{};
        shared.type = QNN_HTP_MEM_SHARED_BUFFER;
        shared.size = bytes;
        shared.sharedBufferConfig = {fd, offset};
        Qnn_MemDescriptor_t memory = QNN_MEM_DESCRIPTOR_INIT;
        memory.memShape = {desc.rank, desc.dimensions, nullptr};
        memory.dataType = desc.dataType;
        memory.memType = QNN_MEM_TYPE_CUSTOM;
        memory.customInfo = &shared;
        const std::string operation = "Register HTP DMA-BUF tensor " + std::string(desc.name);
        handles.push_back(nullptr);
        check(htp.q.memRegister(context, &memory, 1, &handles.back()), operation.c_str());
        desc.memType = QNN_TENSORMEMTYPE_MEMHANDLE;
        desc.memHandle = handles.back();
    }

    void register_tensor(Qnn_Tensor_t& desc, Qnn_ContextHandle_t context, std::size_t offset,
                         std::size_t tensor_bytes) {
        if (desc.version == QNN_TENSOR_VERSION_1)
            register_tensor(desc.v1, context, offset, tensor_bytes);
        else if (desc.version == QNN_TENSOR_VERSION_2)
            register_tensor(desc.v2, context, offset, tensor_bytes);
        else
            throw std::runtime_error("Unsupported QNN tensor descriptor version");
    }
};

inline const char* tensor_name(const Qnn_Tensor_t& tensor) {
    if (tensor.version == QNN_TENSOR_VERSION_1)
        return tensor.v1.name;
    if (tensor.version == QNN_TENSOR_VERSION_2)
        return tensor.v2.name;
    throw std::runtime_error("Unsupported QNN tensor descriptor version");
}

inline unsigned parse_pairs(std::string_view value) {
    unsigned result = 0;
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
    if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() || !result || result > 9999)
        throw std::runtime_error("PAIRS must be an integer in [1, 9999]");
    return result;
}
} // namespace htp_probe
