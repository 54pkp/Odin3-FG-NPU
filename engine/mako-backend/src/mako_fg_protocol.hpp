/* SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
// The socket messages between the Renderer and mako-fg-host, a separate program with its own declaration of this
// format. It is host-endian, and 32-bit Renderers talk to the same 64-bit program, so each message keeps one layout on
// both. Change hostProtocol with any change here; packaging compares it with the program.
#include <cstddef>
#include <cstdint>
namespace mako::backend::fg_host {
constexpr uint32_t hostProtocol=3;
// Local Odin protocol variant. ScheduleAsync has no corresponding reply.
constexpr uint32_t htpVfiAsyncProtocol=0x4F330001;
// The motion variant passes a final memfd in Open. Word zero keeps the
// original batch status. An extended memfd also carries the first phase.
// A status is (output timeline value << 1) | accepted.
constexpr size_t motionBatchWord = 0;
constexpr size_t motionFirstPhaseWord = 1;
constexpr size_t motionStatusWords = 2;
constexpr uint32_t htpVfiMotionProtocol=0x4F330002;
constexpr uint32_t htpVfiTripleProtocol=0x4F330003;
constexpr uint32_t htpVfiLinearOutputProtocol=0x4F330004;
constexpr uint32_t htpVfiLinearIoProtocol=0x4F330005;
constexpr uint32_t htpVfiSharedSourceProtocol=0x4F330007;
// Direct DRI3 adds a fourth shared source descriptor after the third.
// The Renderer ordinal selects one of the three half-phase outputs.
constexpr uint32_t htpVfiDirectDri3Protocol=0x4F330009;
constexpr uint32_t describedSourceProtocol=0x4F33000B;
constexpr uint32_t directDri3OutputCount=3;
constexpr uint32_t directDri3TripleOutputCount=6;
inline constexpr uint32_t directDri3PoolSize(bool triple) {
    return triple ? directDri3TripleOutputCount : directDri3OutputCount;
}
constexpr uint32_t directDri3SourceCount=12;
constexpr int32_t rawLinearOutputEncoding=5;
constexpr int32_t rawLinearInputEncoding=6;
constexpr int32_t rawLinearIoEncoding=7;
constexpr int32_t rawSharedSourceEncoding=8;
constexpr uint32_t sharedSourceCount=3;
constexpr size_t maxOutputs=32;
constexpr size_t maxOpenDescriptors=maxOutputs+18;
inline constexpr uint32_t directDri3OutputSlot(uint64_t rendererOrdinal) {
    return static_cast<uint32_t>(rendererOrdinal % directDri3OutputCount);
}
constexpr uint64_t sharedSourceSlotMask=255;
constexpr unsigned sharedSourceOrdinalShift=8;
inline constexpr uint32_t sourceCount(bool direct) {
    return direct ? directDri3SourceCount : sharedSourceCount;
}
// Direct endpoints fit without the legacy +1 sentinel since A cannot equal B.
inline constexpr uint64_t encodeSourceSchedule(uint32_t a, uint32_t b, uint64_t ordinal, bool direct) {
    const auto slots = a * sourceCount(direct) + b + (direct ? 0 : 1);
    return (ordinal << sharedSourceOrdinalShift) | slots;
}
constexpr int hostDescriptor=3;
enum Op:uint32_t {Hello=1,Devices,Supports,Open,Schedule,Ready,EnableProfiling,ReadProfile,ScheduleAsync,OpenDescribed};
// Schedule's `interval` is the time in seconds between the batch's two real frames, 0 when unknown.
struct alignas(8) Request {
    uint32_t op,count;uint64_t timeout;int32_t device,encoding,period;uint32_t half,width,height;
    float flow,times[maxOutputs],interval;
};
struct Reply {int32_t failed;uint32_t first,second;char text[4084];};
struct GpuInfo {int32_t index;uint32_t vendor,id;char pci[32],name[256];uint8_t driverUuid[16],deviceUuid[16];};
struct alignas(8) ProfileData {
    double previousWait,submit,input,analysis,span,synthesis[maxOutputs],output[maxOutputs],generated[maxOutputs];
    uint32_t validBits,synthesisCount,outputCount,generatedCount;float period;
};
static_assert(sizeof(Request)==176&&sizeof(ProfileData)==832);
}
