import { callable, definePlugin, toaster } from "@decky/api";
import {
  DialogButton,
  Focusable,
  PanelSection,
  PanelSectionRow,
  ToggleField,
  staticClasses,
  useQuickAccessVisible,
} from "@decky/ui";
import { useEffect, useState } from "react";

type Settings = {
  preset: string;
  core_max: number;
  bus_max: number;
  sleep_enable: boolean;
  auto_target_pct: number;
};

type Client = {
  pid: number;
  name: string;
  tier: number;
  core_corner: string;
  core_max: string;
  bus_corner: string;
  sleep_disabled: boolean;
  client_idle: boolean;
  infer_fps: number;
  avg_exec_ms: number;
  p95_exec_ms: number;
  duty_cycle_pct: number;
};

type Status = {
  settings: Settings;
  clients: Client[];
  sensors: {
    hmx_temp_c: number | null;
    hvx_temp_c: number | null;
    ddr_temp_c: number | null;
    sys_power_w: number | null;
    power_source: string;
    battery_pct: number | null;
  };
  services: {
    vela_npu_active: boolean;
    vela_hooked: boolean;
    fg_installed: boolean;
    fg_hooked: boolean;
    shim_installed: boolean;
  };
};

type CustomOpts = { core_max?: number; bus_max?: number; sleep_enable?: boolean };

const getStatus = callable<[], Status>("get_npu_status");
const setProfile = callable<[preset: string, customOpts: CustomOpts | null], Status>("set_npu_profile");
const setVela = callable<[enabled: boolean], Status>("set_vela_service_enabled");

function withTimeout<T>(request: Promise<T>, message: string, ms = 6000): Promise<T> {
  let timer: number | undefined;
  const timeout = new Promise<never>((_, reject) => {
    timer = window.setTimeout(() => reject(new Error(message)), ms);
  });
  return Promise.race([request, timeout]).finally(() => window.clearTimeout(timer));
}

const PRESETS: { id: string; title: string; subtitle: string }[] = [
  { id: "auto", title: "智能动态能效（推荐）", subtitle: "按实时占空比在 SVS+ ~ MAX 间自动升降压，帧间深度休眠" },
  { id: "eco", title: "省电静音档 Eco", subtitle: "低压 SVS 档 · 快速关断 HMX · 适合 720p/60Hz 轻负载" },
  { id: "saver", title: "能效优先档 Saver", subtitle: "核心 NOM+ / 总线 NOM · 明显降低 NPU 与内存功耗" },
  { id: "balanced", title: "均衡甜点档 Balanced", subtitle: "核心 TURBO / 总线 NOM+ · 兼顾 1080p 与功耗" },
  { id: "high", title: "高性能档 High", subtitle: "核心 TURBO+/MAX · 保留帧间睡眠与 HMX 断电" },
  { id: "max", title: "满血极限档 Max", subtitle: "锁定 MAX 电压且禁止休眠（原厂满血行为）" },
  { id: "passthrough", title: "原厂不干预", subtitle: "不修改任何 NPU 电压投票（调节器旁路）" },
];

const CORE_CHOICES: { value: number; label: string }[] = [
  { value: 0x40, label: "SVS" },
  { value: 0x60, label: "NOM" },
  { value: 0x80, label: "TURBO" },
  { value: 0x90, label: "TURBO+" },
  { value: 0xa0, label: "MAX" },
];
const BUS_CHOICES: { value: number; label: string }[] = [
  { value: 0x40, label: "SVS" },
  { value: 0x60, label: "NOM" },
  { value: 0x80, label: "TURBO" },
  { value: 0xa0, label: "MAX" },
];

const HEAD: Record<string, string | number> = { fontSize: "12px", fontWeight: 600, opacity: 0.85, marginBottom: "4px" };

function presetTitle(id: string): string {
  return PRESETS.find((p) => p.id === id)?.title ?? id;
}

function fmt(v: number | null, unit: string): string {
  return v === null || v === undefined ? "--" : `${v}${unit}`;
}

function StatusCard({ status }: { status: Status }) {
  const { clients, sensors, services, settings } = status;
  const live = clients.filter((c) => !c.client_idle);
  const first = live[0] ?? clients[0];
  let dot = "#9ca3af";
  let headline = "NPU 空闲（当前无推理任务）";
  if (settings.preset === "passthrough") {
    dot = "#fb923c";
    headline = "调节器旁路中（原厂满血投票）";
  }
  if (first && !first.client_idle) {
    dot = "#4ade80";
    headline = `${presetTitle(settings.preset)} · Core ${first.core_corner} / Bus ${first.bus_corner}`;
  } else if (first) {
    dot = "#60a5fa";
    headline = "NPU 已释放电压投票（深度休眠）";
  }
  return (
    <div
      style={{ borderRadius: "6px", padding: "10px 12px", background: "rgba(255,255,255,0.06)", fontSize: "12px", lineHeight: "1.55" }}
      aria-live="polite"
    >
      <div style={{ fontSize: "13px", fontWeight: 600 }}>
        <span style={{ color: dot }}>●</span> {headline}
      </div>
      {clients.map((c) => (
        <div key={c.pid} style={{ color: "#86efac", marginTop: "2px" }}>
          {c.name}：{c.infer_fps.toFixed(0)} 次/秒 · 单次 {c.avg_exec_ms.toFixed(1)} ms · 占空比 {c.duty_cycle_pct.toFixed(0)}%
          {c.sleep_disabled ? " · 禁休眠" : " · 可休眠"}
        </div>
      ))}
      <div style={{ opacity: 0.85, marginTop: "4px" }}>
        HMX {fmt(sensors.hmx_temp_c, "°C")} · HVX {fmt(sensors.hvx_temp_c, "°C")} · DDR {fmt(sensors.ddr_temp_c, "°C")}
      </div>
      <div style={{ opacity: 0.85 }}>
        整机 {fmt(sensors.sys_power_w, " W")}（{sensors.power_source}）
        {sensors.battery_pct !== null ? ` · 电量 ${sensors.battery_pct}%` : ""}
      </div>
      <div style={{ opacity: 0.7, marginTop: "4px" }}>
        补帧引擎 {services.fg_installed ? (services.fg_hooked ? "已接管" : "未接管") : "未安装"} · 系统 NPU 守护{" "}
        {services.vela_npu_active ? (services.vela_hooked ? "已接管" : "运行中(未接管)") : "已停止"}
      </div>
    </div>
  );
}

function Content() {
  const visible = useQuickAccessVisible();
  const [status, setStatus] = useState<Status | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!visible) return;
    let stopped = false;
    let timer: number | undefined;
    async function poll() {
      try {
        const s = await withTimeout(getStatus(), "Decky 后端响应超时");
        if (!stopped) {
          setStatus(s);
          setError(null);
        }
      } catch (err) {
        if (!stopped) setError(String(err));
      }
      if (!stopped) timer = window.setTimeout(poll, 1000);
    }
    void poll();
    return () => {
      stopped = true;
      window.clearTimeout(timer);
    };
  }, [visible]);

  async function apply(preset: string, opts: CustomOpts | null, label: string) {
    setBusy(true);
    try {
      const s = await withTimeout(setProfile(preset, opts), "切换档位超时");
      setStatus(s);
      toaster.toast({ title: "Odin3 NPU 性能调节", body: label });
    } catch (err) {
      toaster.toast({ title: "切换失败", body: String(err) });
    } finally {
      setBusy(false);
    }
  }

  async function toggleVela(enabled: boolean) {
    setBusy(true);
    try {
      setStatus(await withTimeout(setVela(enabled), "服务操作超时", 50000));
    } catch (err) {
      toaster.toast({ title: "服务操作失败", body: String(err) });
    } finally {
      setBusy(false);
    }
  }

  if (!status) {
    return (
      <PanelSection title="NPU 性能调节">
        <PanelSectionRow>
          <div style={{ fontSize: "12px" }}>{error ?? "正在读取 NPU 状态…"}</div>
        </PanelSectionRow>
      </PanelSection>
    );
  }

  const s = status.settings;
  const keepAuto = s.preset === "auto";
  const customPreset = keepAuto ? "auto" : "custom";

  return (
    <>
      <PanelSection title="NPU 实时状态">
        <PanelSectionRow>
          <StatusCard status={status} />
        </PanelSectionRow>
      </PanelSection>

      <PanelSection title="性能 / 功耗档位（点击即时生效，无需重启游戏）">
        <PanelSectionRow>
          <Focusable style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
            {PRESETS.map((p) => {
              const active = s.preset === p.id;
              return (
                <DialogButton
                  key={p.id}
                  disabled={busy}
                  onClick={() => {
                    if (!active) void apply(p.id, null, `已切换：${p.title}`);
                  }}
                  style={{
                    minWidth: 0,
                    padding: "8px 12px",
                    textAlign: "left",
                    ...(active ? { background: "#1a9fff", color: "#ffffff" } : {}),
                  }}
                >
                  <div style={{ fontSize: "14px", fontWeight: 600 }}>{active ? `✓ ${p.title}` : p.title}</div>
                  <div style={{ fontSize: "11px", opacity: active ? 0.95 : 0.72, marginTop: "2px" }}>{p.subtitle}</div>
                </DialogButton>
              );
            })}
          </Focusable>
        </PanelSectionRow>
      </PanelSection>

      <PanelSection title="高级微调">
        <PanelSectionRow>
          <div style={HEAD}>
            核心电压上限（{keepAuto ? "限制智能档的最高档位" : s.preset === "custom" ? "自定义档" : "点击将切换为自定义档"}）
          </div>
        </PanelSectionRow>
        <PanelSectionRow>
          <Focusable style={{ display: "flex", flexWrap: "wrap", gap: "6px" }}>
            {CORE_CHOICES.map((c) => {
              const active = (keepAuto || s.preset === "custom") && s.core_max === c.value;
              return (
                <DialogButton
                  key={c.value}
                  disabled={busy}
                  onClick={() => void apply(customPreset, { core_max: c.value }, `核心上限 ${c.label}`)}
                  style={{
                    minWidth: "28%",
                    flex: "1 1 28%",
                    padding: "6px",
                    fontSize: "12px",
                    fontWeight: 600,
                    ...(active ? { background: "#1a9fff", color: "#ffffff" } : {}),
                  }}
                >
                  {active ? `✓ ${c.label}` : c.label}
                </DialogButton>
              );
            })}
          </Focusable>
        </PanelSectionRow>
        <PanelSectionRow>
          <div style={HEAD}>总线（内存带宽）电压上限</div>
        </PanelSectionRow>
        <PanelSectionRow>
          <Focusable style={{ display: "flex", flexWrap: "wrap", gap: "6px" }}>
            {BUS_CHOICES.map((c) => {
              const active = (keepAuto || s.preset === "custom") && s.bus_max === c.value;
              return (
                <DialogButton
                  key={c.value}
                  disabled={busy}
                  onClick={() => void apply(customPreset, { bus_max: c.value }, `总线上限 ${c.label}`)}
                  style={{
                    minWidth: "22%",
                    flex: "1 1 22%",
                    padding: "6px",
                    fontSize: "12px",
                    fontWeight: 600,
                    ...(active ? { background: "#1a9fff", color: "#ffffff" } : {}),
                  }}
                >
                  {active ? `✓ ${c.label}` : c.label}
                </DialogButton>
              );
            })}
          </Focusable>
        </PanelSectionRow>
        <PanelSectionRow>
          <ToggleField
            label="允许帧间深度休眠 / HMX 自动断电"
            description="关闭后 NPU 推理间隙不休眠（仅对自定义档生效，智能档恒为休眠）。"
            checked={s.sleep_enable}
            disabled={busy || keepAuto}
            onChange={(checked: boolean) => void apply("custom", { sleep_enable: checked }, checked ? "已允许帧间休眠" : "已禁止帧间休眠")}
          />
        </PanelSectionRow>
        <PanelSectionRow>
          <ToggleField
            label="后台系统 NPU 守护服务"
            description="vela-npu.service：不使用其他系统 NPU 服务时可关闭，省去空闲 FastRPC 会话。"
            checked={status.services.vela_npu_active}
            disabled={busy}
            onChange={(checked: boolean) => void toggleVela(checked)}
          />
        </PanelSectionRow>
        <PanelSectionRow>
          <div style={{ fontSize: "11px", opacity: 0.72, lineHeight: "1.45" }}>
            作用于 Odin3 补帧引擎与 系统 NPU 守护进程的 NPU（Hexagon HTP v79）电压投票。切换档位对运行中的游戏即时生效。
          </div>
        </PanelSectionRow>
      </PanelSection>
    </>
  );
}

export default definePlugin(() => ({
  name: "Odin3 NPU Power",
  titleView: <div className={staticClasses.Title}>Odin3 NPU 性能调节</div>,
  content: <Content />,
  icon: (
    <span aria-hidden="true" style={{ fontWeight: 700, fontSize: "13px" }}>
      ⚡NPU
    </span>
  ),
}));
