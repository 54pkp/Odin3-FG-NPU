// SPDX-License-Identifier: GPL-3.0-or-later
// Odin3 RawHtpVfiModel: tier-matched native / minimal-sync NEON + HTP hybrid generation (0.00% GPU worker load)
// and dynamic 1-phase (t=0.5) support inside 3x contexts for 54..115 FPS native games.

class RawHtpVfiModel {
public:
    struct PhaseTrace {
        uint64_t source, dispatch;
        uint64_t queryBegin{}, queryEnd{}, recordingBegan{};
        uint64_t inputSubmit{}, inputReady{}, commandResetBegin{}, commandResetEnd{};
        uint64_t readable{}, outputSubmit{}, outputReturned{}, inputCopyBytes{};
        uint64_t rendererOrdinal{};
        unsigned sharedA{}, sharedB{}, lumaMode{};
        unsigned inputCopies{}, outputSlot{};
        odin3_direct_pair::Execution motion{}, phase{};
        LiveHtpProfile::Sample liveProfile{};
        bool profiled{};
    };

private:
    static constexpr size_t traceCapacity = 16384;
    std::vector<PhaseTrace> traces;
    uint64_t untrackedTraces{};

    static uint64_t ns(odin3_direct_pair::Clock::time_point time) {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(time.time_since_epoch()).count();
    }

    void writeTraces() const {
        if (!traces.capacity()) return;
        for (const auto& timing : traces) {
            dprintf(STDERR_FILENO,
                "Odin3 FG worker: pid=%d action=htp-raw-stage-timing tid=%d source=%ju"
                " dispatch_ns=%ju motion_enter_ns=%ju motion_exit_ns=%ju phase_enter_ns=%ju phase_exit_ns=%ju"
                " readable_ns=%ju output_slot=%u\n",
                getpid(), gettid(), uintmax_t(timing.source), uintmax_t(timing.dispatch),
                uintmax_t(ns(timing.motion.start)), uintmax_t(ns(timing.motion.done)),
                uintmax_t(ns(timing.phase.start)), uintmax_t(ns(timing.phase.done)), uintmax_t(timing.readable),
                timing.outputSlot);
        }
        dprintf(STDERR_FILENO,
            "Odin3 FG worker: pid=%d action=htp-raw-stage-summary tid=%d records=%zu untracked=%ju\n",
            getpid(), gettid(), traces.size(), uintmax_t(untrackedTraces));
    }

    struct Buffer {
        VkBuffer handle{};
        VkDeviceMemory memory{};
    };

    const VulkanDevice& gpu;
    odin3_direct_pair::FrameSize frameSize;
    const size_t phaseTotal;
    std::unique_ptr<odin3_direct_pair::Pipeline> pipelineP;
    std::optional<LiveHtpProfile> liveProfile;
    std::array<Buffer, 2> inputs{}, outputs{};
    VkFence transferFence{};
    bool reusePreviousInput{};
    std::optional<uint64_t> lastInputFrame;
    struct SharedPair { unsigned a, b; uint64_t ordinal; };
    std::optional<SharedPair> sharedPair;
    std::optional<SharedPair> currentSharedPair;
    const bool pooledOutput;
    unsigned pooledOutputSlot{};
    struct CapturedPair {
        SharedPair pair;
        unsigned outputSlot;
        uint64_t begin, end;
        std::array<uintmax_t, 3> devices{}, inodes{};
        std::array<std::vector<char>, 3> frames;
    };
    std::filesystem::path captureDirectory;
    uint64_t captureEpoch{};
    std::optional<SharedPair> captureSource;
    std::optional<CapturedPair> captured;

    // ---- Gen-8 tier state ----
    const odin3::shared_frame::DmaSourcePool* describedPtr{};
    std::array<int, 12> keepIn{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
    std::array<int, 2> keepOut{-1, -1};
    std::array<int, 4> keepExtra{-1, -1, -1, -1};
    bool tiered{};
    int nativeIdx{-1};
    int scaleIdx{-1};
    int guardCap{4};
    unsigned pollCounter{};
    odin3_tier::TierState lastTier{};
    unsigned lastBatchPhases{};
    std::unique_ptr<odin3_tier::CpuTierScaler> cpuScaler;
    std::optional<uint64_t> scaledLastB;
    unsigned scaledLastBArena{};
    unsigned scaledLastBSlot{};
    odin3_direct_pair::Clock::time_point pairBegin{}, prevEstimateAt{}, thermalLoweredAt{};
    uint64_t prevPairOrdinal{};
    double pairEmaMs{};
    double sourceIntervalEmaMs{14.0};
    uint64_t fastSourceStreak{};
    uint64_t slowSourceStreak{};
    bool fastSourceLatched{true};
    int thermalCap{4};
    uint32_t lastGuardPreset{999u};
    unsigned overloadPairs{};
    unsigned guardDrops{};
    unsigned guardHoldoff{90};
    odin3_direct_pair::Clock::time_point guardLoweredAt{};
    uint64_t pairCount{};

    // Overlaps HTP `P().render(1)` with CPU NEON `upsampleOutput(0)` during scaled 3x pairs,
    // cutting 3x_eco pair latency from ~13.4 ms down to ~9.3 ms (sustaining 80->240+ FPS even on SVS!).
    std::mutex asyncP1Mtx;
    std::condition_variable asyncP1CvGo, asyncP1CvDone;
    std::thread asyncP1Thread;
    bool asyncP1Stop{false};
    bool asyncP1Pending{false};
    bool asyncP1Ready{false};
    odin3_direct_pair::Execution asyncP1Timing{};
    std::exception_ptr asyncP1Err{};

    void asyncP1Loop() {
        odin3_tier::pinToEfficiencyCluster(scaleIdx == 0);
        for (;;) {
            {
                std::unique_lock lk(asyncP1Mtx);
                asyncP1CvGo.wait(lk, [&] { return asyncP1Stop || asyncP1Pending; });
                if (asyncP1Stop) return;
            }
            odin3_direct_pair::Execution ex{};
            std::exception_ptr ep{};
            try {
                ex = P().render(1);
            } catch (...) {
                ep = std::current_exception();
            }
            {
                std::lock_guard lk(asyncP1Mtx);
                asyncP1Timing = ex;
                asyncP1Err = ep;
                asyncP1Pending = false;
                asyncP1Ready = true;
            }
            asyncP1CvDone.notify_one();
        }
    }

    bool scaledActive() const {
        return tiered && scaleIdx >= 0 && scaleIdx != nativeIdx;
    }

    void publishStatus() noexcept {
        if (!tiered || scaleIdx < 0) return;
        odin3_tier::publishTierStatus(lastTier, scaleIdx, nativeIdx, frameSize.width, frameSize.height,
                                      phaseTotal, lastBatchPhases ? lastBatchPhases : unsigned(phaseTotal),
                                      pairEmaMs, guardCap);
    }

    void capturePixels() noexcept {
        auto pair = std::exchange(captureSource, std::nullopt);
        if (!pair || captured || scaledActive()) return;
        try {
            if (!std::filesystem::remove(captureDirectory / "capture-next")) return;
            CapturedPair result{.pair = *pair, .outputSlot = pooledOutputSlot, .begin = monotonicNs()};
            std::array buffers{&P().input(pair->a), &P().input(pair->b), &P().half_phase_buffer(pooledOutputSlot)};
            for (size_t i = 0; i < buffers.size(); ++i) {
                auto& buffer = *buffers[i];
                struct stat identity{};
                if (fstat(buffer.borrow_fd(), &identity) != 0)
                    throw std::runtime_error("Cannot identify captured DMA-BUF");
                result.devices[i] = static_cast<uintmax_t>(identity.st_dev);
                result.inodes[i] = static_cast<uintmax_t>(identity.st_ino);
                auto& frame = result.frames[i];
                frame.resize(buffer.size());
                buffer.map_cpu();
                buffer.sync(DMA_BUF_SYNC_START | DMA_BUF_SYNC_READ);
                std::memcpy(frame.data(), buffer.at(0), frame.size());
                buffer.sync(DMA_BUF_SYNC_END | DMA_BUF_SYNC_READ);
            }
            result.end = monotonicNs();
            captured.emplace(std::move(result));
        } catch (const std::exception& error) {
            std::fprintf(stderr, "Odin3 diagnostic pair capture failed: %s\n", error.what());
        }
    }

    void savePixels() const noexcept {
        if (!captured) return;
        try {
            const auto& shot = *captured;
            const auto path = captureDirectory / ("pair-" + std::to_string(getpid()) + "-" +
                std::to_string(captureEpoch) + "-" + std::to_string(shot.pair.ordinal));
            if (!std::filesystem::create_directory(path))
                throw std::runtime_error("Capture output already exists");
            const std::array names{"a.bgra", "b.bgra", "m.bgra"};
            for (size_t i = 0; i < names.size(); ++i) {
                const auto& frame = shot.frames[i];
                std::ofstream output(path / names[i], std::ios::binary);
                output.write(frame.data(), static_cast<std::streamsize>(frame.size()));
                output.close();
                if (!output) throw std::runtime_error("Cannot save captured BGRA frame");
            }
        } catch (const std::exception& error) {
            std::fprintf(stderr, "Odin3 diagnostic pair save failed: %s\n", error.what());
        }
    }

    static std::filesystem::path contextPath(const char* variable) {
        const char* value = std::getenv(variable);
        if (!value || !value[0] || value[0] != '/')
            throw std::runtime_error(std::string("Expected an absolute context path in ") + variable);
        return value;
    }

    static std::filesystem::path resolveModelContextPath(const char* variable, odin3_direct_pair::FrameSize frame,
                                                        const char* bundleFile, bool optional = false) {
        const char* value = std::getenv(variable);
        if (value && value[0] == '/') return value;
        const char* bundleDir = std::getenv("ODIN3_FG_MODEL_BUNDLE_DIR");
        if (bundleDir && bundleDir[0] == '/') {
            const char* sub = mako::odin3ModelBundleSubdir(frame.width, frame.height);
            if (!sub)
                throw std::runtime_error("Unsupported swapchain extent for ODIN3_FG_MODEL_BUNDLE_DIR");
            auto candidate = std::filesystem::path(bundleDir) / sub / bundleFile;
            if (std::filesystem::is_regular_file(candidate)) return candidate;
            if (optional) return {};
            throw std::runtime_error("Missing pure-neural context binary: " + candidate.string());
        }
        if (optional) return {};
        return contextPath(variable);
    }

    static std::pair<std::filesystem::path, std::filesystem::path> lumaContextPaths() {
        if (!std::getenv("ODIN3_HTP_RAW_LUMA_COLD") && !std::getenv("ODIN3_HTP_RAW_LUMA_WARM"))
            return {};
        return {contextPath("ODIN3_HTP_RAW_LUMA_COLD"), contextPath("ODIN3_HTP_RAW_LUMA_WARM")};
    }

    static bool topPowerRequested() {
        const char* value = std::getenv("ODIN3_HTP_PROBE_TOP_POWER");
        return value && std::string_view(value) == "1";
    }

    void import(Buffer& buffer, const htp_probe::DmaArena& arena, VkBufferUsageFlags usage) const {
        const int fd = fcntl(arena.borrow_fd(), F_DUPFD_CLOEXEC, 0);
        check(fd >= 0, "htp-raw-dmabuf-dup");
        importHtpVfiDmaBuffer(gpu, fd, arena.size(), usage, true, buffer.handle, buffer.memory);
    }

    odin3_direct_pair::Pipeline& P() { return *pipelineP; }
    const odin3_direct_pair::Pipeline& P() const { return *pipelineP; }

    // Builds the pipeline for one internal scale. `idx == nativeIdx` binds the shared full-resolution DMA-BUFs directly.
    std::unique_ptr<odin3_direct_pair::Pipeline> makePipeline(int idx) {
        using odin3_direct_pair::Pipeline;
        namespace fs = std::filesystem;
        const bool uncappedGameCompact = (idx == 0 && !isCappedSlowSource() && std::getenv("ODIN3_FG_RUNTIME_APP_ID") != nullptr);
        ::setenv("ODIN3_NPU_UNCAPPED_COMPACT", uncappedGameCompact ? (phaseTotal == 2 ? "2" : "1") : "0", 1);
        const char* bundle = std::getenv("ODIN3_FG_MODEL_BUNDLE_DIR");
        const auto path = [&](const odin3_tier::Scale& s, const char* file) { return fs::path(bundle) / s.sub / file; };
        const auto& s = odin3_tier::kScales.at(size_t(idx));
        const bool native = idx == nativeIdx;
        const odin3_direct_pair::FrameSize size{s.w, s.h};
        const fs::path halfInTriple = phaseTotal == 2 ? path(s, "phase12.bin") : fs::path{};
        std::unique_ptr<Pipeline> p;
        if (native) {
            const bool joined = phaseTotal == 1 && fs::is_regular_file(path(s, "vfi-joined.bin")) &&
                (idx == 4 || !fs::is_regular_file(path(s, "phase12.bin")));
            const std::array<int, 2> outFds = keepOut;
            const std::array<int, 12> inFds = keepIn;
            if (joined) {
                p = std::make_unique<Pipeline>(fs::path{}, path(s, "vfi-joined.bin"), fs::path{}, topPowerRequested(), false,
                    odin3_direct_pair::configured_analysis_stride(), outFds, inFds, std::getenv("ODIN3_HTP_RAW_PHASE_OP_PACKAGE"),
                    std::pair<fs::path, fs::path>{}, size, true, describedPtr);
            } else {
                p = std::make_unique<Pipeline>(path(s, "vfi-motion3x.bin"),
                    phaseTotal == 2 ? path(s, "phase13.bin") : path(s, "phase12.bin"),
                    phaseTotal == 2 ? path(s, "phase23.bin") : fs::path{}, topPowerRequested(), false,
                    odin3_direct_pair::configured_analysis_stride(), outFds, inFds, std::getenv("ODIN3_HTP_RAW_PHASE_OP_PACKAGE"),
                    std::pair<fs::path, fs::path>{}, size, false, describedPtr, halfInTriple);
            }
            if (pooledOutput) {
                if (phaseTotal == 1) {
                    p->register_half_phase_output_slots({keepExtra[0], keepExtra[1]});
                } else {
                    p->register_triple_phase_output_slots(keepExtra);
                }
            }
        } else {
            // map_for_cpu = true: compact sub-resolution HTP DMA-BUFs are resampled by minimal-sync Oryon NEON (0.00% GPU load)
            p = std::make_unique<Pipeline>(path(s, "vfi-motion3x.bin"),
                phaseTotal == 2 ? path(s, "phase13.bin") : path(s, "phase12.bin"),
                phaseTotal == 2 ? path(s, "phase23.bin") : fs::path{}, topPowerRequested(), true,
                odin3_direct_pair::configured_analysis_stride(), std::array<int, 2>{-1, -1},
                std::array<int, 12>{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, std::getenv("ODIN3_HTP_RAW_PHASE_OP_PACKAGE"),
                std::pair<fs::path, fs::path>{}, size, false, nullptr, halfInTriple);
            odin3_tier::invalidateRangeEl0(p->input(0).at(0), p->input(0).size());
            odin3_tier::invalidateRangeEl0(p->input(1).at(0), p->input(1).size());
        }
        return p;
    }

    // Releases the current FastRPC/HTP device before opening the next scale so two HTP sessions never coexist.
    void retarget(int wanted) {
        if (wanted == scaleIdx && pipelineP) return;
        const int prevIdx = scaleIdx >= 0 ? scaleIdx : nativeIdx;
        liveProfile.reset();
        pipelineP.reset();
        try {
            pipelineP = makePipeline(wanted);
            scaleIdx = wanted;
            odin3_tier::pinToEfficiencyCluster(wanted == 0);
            if (scaledActive() && cpuScaler) {
                const auto& sc = odin3_tier::kScales.at(size_t(wanted));
                cpuScaler->configureScale(frameSize.width, frameSize.height, sc.w, sc.h);
            }
            scaledLastB.reset();
            pairEmaMs = 0;
            overloadPairs = 0;
            guardHoldoff = 60;
            if (const char* profile = std::getenv("ODIN3_FG_LIVE_HTP_PROFILE"); profile && std::string_view(profile) == "1")
                liveProfile.emplace(P().live_profile_backend());
            publishStatus();
            dprintf(STDERR_FILENO, "Odin3 FG worker: pid=%d action=odin3-tier-scale preset=%s scale=%s native=%d phases=%zu\n",
                getpid(), odin3_tier::presetName(lastTier.preset), odin3_tier::kScales.at(size_t(wanted)).sub,
                wanted == nativeIdx, phaseTotal);
        } catch (const std::exception& error) {
            dprintf(STDERR_FILENO, "Odin3 FG worker: pid=%d action=odin3-tier-retarget-failed wanted=%d error=%s\n",
                getpid(), wanted, error.what());
            guardCap = std::max(0, prevIdx);
            pipelineP = makePipeline(prevIdx);
            scaleIdx = prevIdx;
            scaledLastB.reset();
            publishStatus();
        }
    }

    static bool isCappedSlowSource() noexcept {
        if (std::getenv("ODIN3_FG_DISABLE_FAST_SOURCE_CAP")) return true;
        const char* cap = std::getenv("MAKO_BASE_FPS_CAP");
        if (!cap || !cap[0]) return false;
        const int v = std::atoi(cap);
        return v > 0 && v <= 62;
    }

    static bool useStrictHudMode() noexcept {
        static const bool envStrict = [] {
            const char* s = std::getenv("ODIN3_FG_STRICT_HUD");
            return s && s[0] == '1';
        }();
        return envStrict || isCappedSlowSource();
    }

    int desiredScale() {
        if (isCappedSlowSource()) fastSourceLatched = false;
        lastTier = odin3_tier::readTier();
        if (lastTier.preset != lastGuardPreset) {
            lastGuardPreset = lastTier.preset;
            guardCap = 4;
            overloadPairs = 0;
            guardDrops = 0;
            if (odin3_tier::readHmxTempMc() < 80000) thermalCap = 4;
        }
        if (guardCap < 4 && guardDrops == 1 && pairEmaMs > 0.0 && pairEmaMs < 10.0 &&
                odin3_direct_pair::Clock::now() - guardLoweredAt > std::chrono::seconds(20)) {
            ++guardCap;
        }
        if (const int forced = odin3_tier::forcedScaleIndex(); forced >= 0) return std::min(forced, nativeIdx);
        const size_t effPhases = (lastBatchPhases >= 1 && lastBatchPhases <= phaseTotal) ? size_t(lastBatchPhases) : phaseTotal;
        int idx = std::min(odin3_tier::tierCeilIndex(lastTier, effPhases, nativeIdx), nativeIdx);
        const bool liveGame = std::getenv("ODIN3_FG_RUNTIME_APP_ID") != nullptr;
        // When uncapped (`MAKO_BASE_FPS_CAP` unset) on 1080p, run compact 360p so native FPS stays >= 97% with 0 stalls;
        // when capped (`<=62` FPS for 120 Hz lock), use full tier-matched scales (900p on max/high, 720p on auto/balanced,
        // 540p on saver, 360p on eco)!
        if (nativeIdx >= 4 && !isCappedSlowSource()) {
            idx = 0;
        }
        // Proactive HMX thermal guard (latched during live gameplay so FastRPC contexts never ping-pong mid-session!):
        const int hmxMc = odin3_tier::readHmxTempMc();
        const auto nowTh = odin3_direct_pair::Clock::now();
        if (hmxMc >= 90500 && thermalCap > 0) {
            thermalCap = 0;
            thermalLoweredAt = nowTh;
        } else if (hmxMc >= 88000 && thermalCap > 1) {
            thermalCap = 1;
            thermalLoweredAt = nowTh;
        } else if (hmxMc >= 85000 && thermalCap > 2) {
            thermalCap = 2;
            thermalLoweredAt = nowTh;
        } else if (!liveGame && thermalCap < 4 && nowTh - thermalLoweredAt > std::chrono::seconds(15)) {
            if (thermalCap == 0 && hmxMc <= 75000) { thermalCap = 1; thermalLoweredAt = nowTh; }
            else if (thermalCap <= 2 && hmxMc <= 71000) { thermalCap = 4; thermalLoweredAt = nowTh; }
        }
        idx = std::min(idx, thermalCap);
        if (guardCap < idx) {
            idx = std::min(idx, guardCap);
        }
        return std::max(0, idx);
    }

    void prepareScaledInputs(const SharedPair& pair) {
        const bool strictHud = useStrictHudMode();
        const auto& s = odin3_tier::kScales.at(size_t(scaleIdx));
        const bool reuseA = scaledLastB && pair.ordinal > 0 && *scaledLastB == pair.ordinal - 1 && scaledLastBSlot == pair.a;
        const unsigned aArena = reuseA ? scaledLastBArena : 0u;
        const unsigned bArena = 1u - aArena;

        if (!reuseA) {
            if (strictHud)
                cpuScaler->syncInputsPair(pair.a, pair.ordinal * 2ull, pair.b, pair.ordinal * 2ull + 1ull);
            auto& dstA = P().input(aArena);
            cpuScaler->downsampleTo(pair.a, reinterpret_cast<uint8_t*>(dstA.at(0)), dstA.borrow_fd(),
                                    frameSize.width, frameSize.height, s.w, s.h, strictHud);
        } else if (strictHud) {
            cpuScaler->syncInputOnce(pair.b, pair.ordinal * 2ull + 1ull);
        }
        auto& dstB = P().input(bArena);
        cpuScaler->downsampleTo(pair.b, reinterpret_cast<uint8_t*>(dstB.at(0)), dstB.borrow_fd(),
                                frameSize.width, frameSize.height, s.w, s.h, strictHud);

        P().set_input_slots(aArena, bArena);
        scaledLastB = pair.ordinal;
        scaledLastBArena = bArena;
        scaledLastBSlot = pair.b;
    }

    void upsampleOutput(unsigned phase, bool midpointInTriple) {
        if (!currentSharedPair) return;
        const auto& s = odin3_tier::kScales.at(size_t(scaleIdx));
        const unsigned group = pooledOutput ? pooledOutputSlot : 0u;
        const size_t outIdx = size_t(group) * 2 + (phaseTotal == 2 ? phase : 0u);
        const unsigned lowPhase = midpointInTriple ? 0u : phase;
        const uint32_t tq8 = (phaseTotal == 1 || midpointInTriple) ? 128u : (phase == 0 ? 85u : 171u);
        auto& lowBuf = P().output(lowPhase);
        cpuScaler->upsamplePhase(currentSharedPair->a, currentSharedPair->b,
                                 reinterpret_cast<const uint8_t*>(lowBuf.at(0)), lowBuf.borrow_fd(),
                                 outIdx, frameSize.width, frameSize.height, s.w, s.h, tq8, useStrictHudMode(), lastTier.preset);
    }

    void closeKept() {
        for (int& fd : keepIn) if (fd >= 0) { ::close(fd); fd = -1; }
        for (int& fd : keepOut) if (fd >= 0) { ::close(fd); fd = -1; }
        for (int& fd : keepExtra) if (fd >= 0) { ::close(fd); fd = -1; }
    }

public:
    RawHtpVfiModel(const VulkanDevice& device, size_t phases, std::array<int, 2> outputFds = {-1, -1},
                  std::array<int, 12> inputFds = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                  std::array<int, 4> extraOutputFds = {-1, -1, -1, -1},
                  odin3_direct_pair::FrameSize frame = {}, bool joined = false,
                  const odin3::shared_frame::DmaSourcePool* describedSources = nullptr)
        : gpu(device), frameSize(frame), phaseTotal(phases), pooledOutput(extraOutputFds[0] >= 0) {
        odin3_tier::pinToEfficiencyCluster();
        describedPtr = describedSources;
        for (size_t i = 0; i < inputFds.size(); ++i)
            if (inputFds[i] >= 0) keepIn[i] = fcntl(inputFds[i], F_DUPFD_CLOEXEC, 0);
        for (size_t i = 0; i < outputFds.size(); ++i)
            if (outputFds[i] >= 0) keepOut[i] = fcntl(outputFds[i], F_DUPFD_CLOEXEC, 0);
        for (size_t i = 0; i < extraOutputFds.size(); ++i)
            if (extraOutputFds[i] >= 0) keepExtra[i] = fcntl(extraOutputFds[i], F_DUPFD_CLOEXEC, 0);

        const char* bundle = std::getenv("ODIN3_FG_MODEL_BUNDLE_DIR");
        const bool explicitCtx = (std::getenv("ODIN3_HTP_RAW_JOINED_CONTEXT") && std::getenv("ODIN3_HTP_RAW_JOINED_CONTEXT")[0] == '/') ||
                                 (std::getenv("ODIN3_HTP_RAW_FIRST_CONTEXT") && std::getenv("ODIN3_HTP_RAW_FIRST_CONTEXT")[0] == '/');
        nativeIdx = odin3_tier::nativeIndex(frame.width, frame.height);
        tiered = bundle && bundle[0] == '/' && !explicitCtx && nativeIdx >= 0 && pooledOutput && inputFds[2] >= 0 &&
                 !std::getenv("ODIN3_FG_DISABLE_TIERS");
        if (tiered) {
            const size_t fullBytes = size_t(frame.width) * frame.height * 4;
            cpuScaler = std::make_unique<odin3_tier::CpuTierScaler>();
            cpuScaler->mapFullBuffers(keepIn, keepOut, keepExtra, fullBytes, phases);
            lastBatchPhases = unsigned(phases);
            scaleIdx = desiredScale();
            odin3_tier::pinToEfficiencyCluster(scaleIdx == 0);
            pipelineP = makePipeline(scaleIdx);
            if (scaledActive()) {
                const auto& sc = odin3_tier::kScales.at(size_t(scaleIdx));
                cpuScaler->configureScale(frame.width, frame.height, sc.w, sc.h);
            }
            if (phases == 2) {
                asyncP1Thread = std::thread([this] { asyncP1Loop(); });
            }
            publishStatus();
            dprintf(STDERR_FILENO, "Odin3 FG worker: pid=%d action=odin3-tier-scale preset=%s scale=%s native=%d phases=%zu initial=1\n",
                getpid(), odin3_tier::presetName(lastTier.preset), odin3_tier::kScales.at(size_t(scaleIdx)).sub,
                scaleIdx == nativeIdx, phases);
        } else {
            const auto halfInTriple = (!joined && phases == 2)
                ? resolveModelContextPath("ODIN3_HTP_RAW_HALF_CONTEXT", frame, "phase12.bin", true)
                : std::filesystem::path{};
            pipelineP = std::make_unique<odin3_direct_pair::Pipeline>(
                joined ? std::filesystem::path{} : resolveModelContextPath("ODIN3_HTP_RAW_MOTION_CONTEXT", frame, "vfi-motion3x.bin", true),
                joined && phases == 1
                    ? resolveModelContextPath("ODIN3_HTP_RAW_JOINED_CONTEXT", frame, "vfi-joined.bin")
                    : resolveModelContextPath("ODIN3_HTP_RAW_FIRST_CONTEXT", frame, "phase13.bin"),
                phases == 2 ? resolveModelContextPath("ODIN3_HTP_RAW_SECOND_CONTEXT", frame, "phase23.bin") : std::filesystem::path{},
                topPowerRequested(), false, odin3_direct_pair::configured_analysis_stride(), outputFds, inputFds,
                std::getenv("ODIN3_HTP_RAW_PHASE_OP_PACKAGE"),
                joined ? std::pair<std::filesystem::path, std::filesystem::path>{} : lumaContextPaths(), frame, joined,
                describedSources, halfInTriple);
            if (pooledOutput) {
                if (phases == 1) P().register_half_phase_output_slots({extraOutputFds[0], extraOutputFds[1]});
                else P().register_triple_phase_output_slots(extraOutputFds);
            }
        }
        check(phases == P().phase_count(), "htp-raw-context-count");
        check(pooledOutput == (extraOutputFds[1] >= 0), "htp-incomplete-output-pool");
        if (pooledOutput) {
            check(outputFds[0] >= 0 && inputFds[2] >= 0, "htp-invalid-output-pool");
            if (phases == 1) check(extraOutputFds[2] < 0 && extraOutputFds[3] < 0, "htp-unexpected-triple-pool");
            else check(outputFds[1] >= 0 && extraOutputFds[2] >= 0 && extraOutputFds[3] >= 0, "htp-incomplete-triple-output-pool");
            const char* profile = std::getenv("ODIN3_FG_LIVE_HTP_PROFILE");
            if (profile && std::string_view(profile) == "1") liveProfile.emplace(P().live_profile_backend());
            const char* directory = std::getenv("ODIN3_HTP_PAIR_CAPTURE_DIR");
            if (directory && directory[0] == '/' && describedSources) {
                captureDirectory = directory;
                captureEpoch = describedSources->epoch;
            }
        }
        for (unsigned i = 0; i < inputs.size(); ++i)
            if (inputFds[i] < 0 && !scaledActive())
                import(inputs[i], P().input(i), VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        for (unsigned i = 0; i < phases; ++i)
            if (outputFds[i] < 0 && !scaledActive())
                import(outputs[i], P().output(i), VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
        const VkFenceCreateInfo info{.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        check(vkCreateFence(gpu.device, &info, nullptr, &transferFence) == VK_SUCCESS, "htp-raw-fence-create");
        const auto* reuse = std::getenv("ODIN3_HTP_RAW_REUSE_PREVIOUS");
        reusePreviousInput = reuse && std::string_view(reuse) == "1";
        if (reusePreviousInput && !scaledActive()) P().set_input_order(false);
        const auto* trace = std::getenv("ODIN3_HTP_STAGE_TIMING");
        if (trace && std::string_view(trace) == "1") traces.reserve(traceCapacity);
        event("htp-raw-htp-ready", phases);
    }

    ~RawHtpVfiModel() {
        if (asyncP1Thread.joinable()) {
            {
                std::lock_guard lk(asyncP1Mtx);
                asyncP1Stop = true;
            }
            asyncP1CvGo.notify_one();
            asyncP1Thread.join();
        }
        savePixels();
        writeTraces();
        vkDestroyFence(gpu.device, transferFence, nullptr);
        for (const auto buffer : {inputs[0], inputs[1], outputs[0], outputs[1]}) {
            if (buffer.handle) vkDestroyBuffer(gpu.device, buffer.handle, nullptr);
            if (buffer.memory) vkFreeMemory(gpu.device, buffer.memory, nullptr);
        }
        liveProfile.reset();
        pipelineP.reset();
        cpuScaler.reset();
        closeKept();
    }

    PhaseTrace* beginPhaseTrace(uint64_t source) {
        if (!traces.capacity()) return nullptr;
        if (traces.size() == traceCapacity) {
            ++untrackedTraces;
            return nullptr;
        }
        return &traces.emplace_back(PhaseTrace{.source = source, .dispatch = monotonicNs()});
    }

    struct InputPlan {
        unsigned a, b;
        bool reuseA;
    };

    InputPlan planInputs(uint64_t frame) {
        if (!reusePreviousInput) return {0, 1, false};
        const auto b = static_cast<unsigned>(frame % inputs.size());
        const auto a = 1 - b;
        P().set_input_order(a != 0);
        return {a, b, lastInputFrame && frame > *lastInputFrame && frame - *lastInputFrame == 1};
    }

    void markInputsReady(uint64_t frame) {
        if (reusePreviousInput) lastInputFrame = frame;
    }

    void bindLinearInputs(uint64_t frame) {
        if (!scaledActive()) P().set_input_order((frame % inputs.size()) == 0);
    }
    void bindSharedInputs(unsigned a, unsigned b, uint64_t ordinal) {
        sharedPair = SharedPair{a, b, ordinal};
        currentSharedPair = sharedPair;
        if (pooledOutput) pooledOutputSlot = directDri3OutputSlot(ordinal);
        if (!scaledActive()) P().set_input_slots(a, b);
    }

    void discardLumaHistory() { P().discard_luma_history(); }

    unsigned phaseCount() const { return unsigned(phaseTotal); }

    VkBuffer input(unsigned index) const { return inputs.at(index).handle; }
    VkBuffer output(unsigned index) const { return outputs.at(index).handle; }

    void submitInputs(VkCommandBuffer command, VkSemaphore timeline, uint64_t ready) const {
        check(vkEndCommandBuffer(command) == VK_SUCCESS &&
              vkResetFences(gpu.device, 1, &transferFence) == VK_SUCCESS, "htp-raw-input-commands");
        const VkTimelineSemaphoreSubmitInfo values{.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
            .waitSemaphoreValueCount = 1, .pWaitSemaphoreValues = &ready};
        const VkPipelineStageFlags stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
        const VkSubmitInfo submit{.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .pNext = &values,
            .waitSemaphoreCount = 1, .pWaitSemaphores = &timeline, .pWaitDstStageMask = &stage,
            .commandBufferCount = 1, .pCommandBuffers = &command};
        {
            const std::lock_guard lock(gpu.queueLock);
            check(vkQueueSubmit(gpu.queue, 1, &submit, transferFence) == VK_SUCCESS, "htp-raw-input-submit");
        }
        check(vkWaitForFences(gpu.device, 1, &transferFence, VK_TRUE, 2000000000ULL) == VK_SUCCESS,
              "htp-raw-input-timeout");
    }

    void estimate(uint64_t source, PhaseTrace* trace) {
        static const bool cappedSlowSource = [] {
            const char* cap = std::getenv("MAKO_BASE_FPS_CAP");
            if (!cap || !cap[0]) return false;
            const int v = std::atoi(cap);
            return v > 0 && v <= 62;
        }();
        const auto nowEst = odin3_direct_pair::Clock::now();
        const auto pair = std::exchange(sharedPair, std::nullopt);
        if (cappedSlowSource) {
            fastSourceLatched = false;
            fastSourceStreak = 0;
        } else if (pair && prevEstimateAt != odin3_direct_pair::Clock::time_point{} && pair->ordinal > prevPairOrdinal &&
                   !std::getenv("ODIN3_FG_DISABLE_FAST_SOURCE_CAP")) {
            const uint64_t diff = pair->ordinal - prevPairOrdinal;
            if (diff >= 1 && diff <= 4) {
                const double dtFrameMs = std::chrono::duration<double, std::milli>(nowEst - prevEstimateAt).count() / static_cast<double>(diff);
                if (dtFrameMs >= 2.0 && dtFrameMs <= 50.0) {
                    sourceIntervalEmaMs = sourceIntervalEmaMs * 0.80 + dtFrameMs * 0.20;
                    if (sourceIntervalEmaMs < 15.8) {
                        slowSourceStreak = 0;
                        if (++fastSourceStreak >= 8) fastSourceLatched = true;
                    } else if (sourceIntervalEmaMs > 17.2) {
                        fastSourceStreak = 0;
                        if (++slowSourceStreak >= 24) fastSourceLatched = false;
                    }
                }
            }
        }
        if (pair) {
            prevEstimateAt = nowEst;
            prevPairOrdinal = pair->ordinal;
        }
        const unsigned pollMod = (isCappedSlowSource() || !std::getenv("ODIN3_FG_RUNTIME_APP_ID")) ? 8u : 32u;
        if (tiered && (pollCounter++ % pollMod == 0)) {
            const int prevScale = scaleIdx;
            retarget(desiredScale());
            if (scaleIdx != prevScale) {
                if (!scaledActive() && pair) P().set_input_slots(pair->a, pair->b);
                // Reset `prevEstimateAt` after QNN context reload so reload latency never resets `fastSourceStreak`!
                if (pair) prevEstimateAt = odin3_direct_pair::Clock::now();
            }
        }
        // Start pair timer AFTER any one-time tier retarget so context-load latency never triggers guardPair!
        pairBegin = odin3_direct_pair::Clock::now();
        if (tiered && scaledActive() && pair) prepareScaledInputs(*pair);
        auto work = std::pair{odin3_direct_pair::Execution{}, 0u};
        if (P().has_separate_motion())
            work = (pair && !scaledActive()) ? P().estimate_shared(pair->a, pair->b, pair->ordinal) : std::pair{P().estimate(), 0u};
        const auto& [timing, luma] = work;
        if (trace) {
            trace->motion = timing;
            trace->lumaMode = luma;
            if (pair) {
                trace->rendererOrdinal = pair->ordinal;
                trace->sharedA = pair->a;
                trace->sharedB = pair->b;
            }
        }
        if (!captureDirectory.empty() && !captured) captureSource = pair;
        if (P().has_separate_motion())
            timingLog({"htp-raw-motion", source, 0, timing.duration_us()});
    }

    void collectLiveProfile(PhaseTrace& timing) const noexcept {
        if (timing.profiled) liveProfile->collect(timing.liveProfile);
    }

    void render(unsigned phase, uint64_t source, PhaseTrace* trace) {
        render(phase, unsigned(phaseTotal), source, trace);
    }

    void render(unsigned phase, unsigned batchPhases, uint64_t source, PhaseTrace* trace) {
        auto profile = trace && liveProfile ? liveProfile->get() : nullptr;
        const bool midpointInTriple = (phaseTotal == 2 && batchPhases == 1);
        odin3_direct_pair::Execution timing;
        if (scaledActive()) {
            if (batchPhases == 2 && isCappedSlowSource() && asyncP1Thread.joinable()) {
                if (phase == 0) {
                    timing = P().render(0);
                    {
                        std::lock_guard lk(asyncP1Mtx);
                        asyncP1Ready = false;
                        asyncP1Err = nullptr;
                        asyncP1Pending = true;
                    }
                    asyncP1CvGo.notify_one();
                    upsampleOutput(0, false);
                } else {
                    std::unique_lock lk(asyncP1Mtx);
                    asyncP1CvDone.wait(lk, [&] { return asyncP1Ready; });
                    asyncP1Ready = false;
                    auto ep = asyncP1Err;
                    timing = asyncP1Timing;
                    lk.unlock();
                    if (ep) std::rethrow_exception(ep);
                    upsampleOutput(1, false);
                }
            } else {
                timing = midpointInTriple ? P().render_midpoint() : P().render(phase);
                upsampleOutput(phase, midpointInTriple);
            }
        } else if (midpointInTriple) {
            timing = pooledOutput ? P().render_midpoint_to_group(pooledOutputSlot, profile) : P().render_midpoint();
        } else {
            timing = pooledOutput ? P().render_phase_to_group(phase, pooledOutputSlot, profile) : P().render(phase);
        }
        if (trace) {
            trace->phase = timing;
            trace->profiled = profile != nullptr;
            trace->outputSlot = pooledOutput ? (phaseTotal == 2 ? pooledOutputSlot * 2 + phase : pooledOutputSlot) : phase;
            if (phase > 0 && currentSharedPair) {
                trace->rendererOrdinal = currentSharedPair->ordinal;
                trace->sharedA = currentSharedPair->a;
                trace->sharedB = currentSharedPair->b;
            }
        }
        if (!captureDirectory.empty()) capturePixels();
        timingLog({"htp-raw-phase", source, 0, timing.duration_us()});
        if (tiered && phase + 1 == batchPhases) guardPair(batchPhases);
    }

    // Deadline guard: a scale whose whole pair (NEON resample + NPU) stays above budget is lowered one step.
    void guardPair(unsigned batchPhases) {
        const double ms = std::chrono::duration<double, std::milli>(odin3_direct_pair::Clock::now() - pairBegin).count();
        ++pairCount;
        lastBatchPhases = batchPhases;
        if (guardHoldoff > 0) {
            if (--guardHoldoff == 0) pairEmaMs = ms;
            if (pairCount % 60 == 0) publishStatus();
            return;
        }
        pairEmaMs = pairEmaMs == 0 ? ms : pairEmaMs * 0.9 + ms * 0.1;
        const double budget = batchPhases == 2 ? 24.4 : (fastSourceLatched ? 14.6 : 16.40);
        const unsigned overloadLimit = batchPhases == 2 ? 16u : 36u;
        if (pairEmaMs > budget && ms > budget - 0.6 && scaleIdx > 0) {
            if (++overloadPairs >= overloadLimit) {
                guardCap = scaleIdx - 1;
                ++guardDrops;
                guardLoweredAt = odin3_direct_pair::Clock::now();
                overloadPairs = 0;
                dprintf(STDERR_FILENO, "Odin3 FG worker: pid=%d action=odin3-tier-guard lowered_to=%s ema_ms=%.2f\n",
                    getpid(), odin3_tier::kScales.at(size_t(guardCap)).sub, pairEmaMs);
                publishStatus();
            }
        } else {
            overloadPairs = (overloadPairs > 2u) ? (overloadPairs - 2u) : 0u;
        }
        if (pairCount % 60 == 0) publishStatus();
        if (verboseWorkerLogging() && pairCount % 120 == 0)
            dprintf(STDERR_FILENO, "Odin3 FG worker: pid=%d action=odin3-tier-pair scale=%s ema_ms=%.2f\n", getpid(),
                odin3_tier::kScales.at(size_t(std::max(0, scaleIdx))).sub, pairEmaMs);
    }
};
