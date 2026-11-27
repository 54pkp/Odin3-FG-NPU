#pragma once

#include "HTP/QnnHtpProfile.h"

class LiveHtpProfile {
    const QNN_INTERFACE_VER_TYPE& api;
    using Handle = std::unique_ptr<void, QnnProfile_FreeFn_t>;
    Handle profile;

public:
    static constexpr std::array<QnnProfile_EventType_t, 7> types{
        QNN_HTP_PROFILE_EVENTTYPE_GRAPH_EXECUTE_HOST_RPC_TIME_MICROSEC,
        QNN_HTP_PROFILE_EVENTTYPE_GRAPH_EXECUTE_HTP_RPC_TIME_MICROSEC,
        QNN_HTP_PROFILE_EVENTTYPE_GRAPH_EXECUTE_ACCEL_TIME_MICROSEC,
        QNN_HTP_PROFILE_EVENTTYPE_GRAPH_EXECUTE_ACCEL_EXCL_WAIT_TIME_MICROSEC,
        QNN_HTP_PROFILE_EVENTTYPE_GRAPH_EXECUTE_YIELD_INSTANCE_WAIT_TIME,
        QNN_HTP_PROFILE_EVENTTYPE_GRAPH_EXECUTE_VTCM_ACQUIRE_TIME,
        QNN_HTP_PROFILE_EVENTTYPE_GRAPH_EXECUTE_RESOURCE_POWER_UP_TIME,
    };
    struct Sample {
        std::array<uint64_t, types.size()> us{};
        unsigned present{}, invalid{};
        uint64_t collectNs{};
    };

private:
    bool visit(const QnnProfile_EventId_t* ids, uint32_t count, unsigned depth,
               unsigned& budget, Sample& sample) const noexcept {
        if ((count && !ids) || count > budget || depth > 8) return false;
        budget -= count;
        for (uint32_t n = 0; n < count; ++n) {
            QnnProfile_EventData_t event = QNN_PROFILE_EVENT_DATA_INIT;
            if (api.profileGetEventData(ids[n], &event) != QNN_SUCCESS) return false;
            for (unsigned i = 0; i < types.size(); ++i) {
                if (event.type != types[i]) continue;
                const unsigned bit = 1u << i;
                if ((sample.present & bit) || event.unit != QNN_PROFILE_EVENTUNIT_MICROSEC)
                    sample.invalid |= bit;
                sample.present |= bit;
                sample.us[i] = event.value;
                break;
            }
            const QnnProfile_EventId_t* child = nullptr;
            uint32_t children = 0;
            if (api.profileGetSubEvents(ids[n], &child, &children) != QNN_SUCCESS ||
                (children && !visit(child, children, depth + 1, budget, sample))) return false;
        }
        return true;
    }

public:
    explicit LiveHtpProfile(const htp_probe::Htp& backend)
        : api(backend.q), profile(nullptr, api.profileFree) {
        if (!api.profileCreate || !api.profileFree || !api.profileGetEvents ||
            !api.profileGetSubEvents || !api.profileGetEventData)
            throw std::runtime_error("The HTP provider lacks basic profiling APIs");
        Qnn_ProfileHandle_t handle = nullptr;
        const auto status = api.profileCreate(backend.backend, QNN_PROFILE_LEVEL_BASIC, &handle);
        profile.reset(handle);
        htp_probe::check(status, "Create HTP live execution profile");
        if (!profile) throw std::runtime_error("The HTP provider returned a null profile");
    }

    Qnn_ProfileHandle_t get() const noexcept { return profile.get(); }

    void collect(Sample& sample) const noexcept {
        const uint64_t start = monotonicNs();
        const QnnProfile_EventId_t* events = nullptr;
        uint32_t count = 0;
        unsigned budget = 256;
        if (api.profileGetEvents(get(), &events, &count) != QNN_SUCCESS || !count ||
            !visit(events, count, 0, budget, sample)) sample.invalid |= 1u << 31;
        sample.collectNs = monotonicNs() - start;
    }
};
