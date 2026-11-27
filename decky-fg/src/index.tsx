import { callable, definePlugin, toaster } from "@decky/api";
import {
  DialogButton,
  Focusable,
  PanelSection,
  PanelSectionRow,
  Router,
  ToggleField,
  staticClasses,
  useQuickAccessVisible,
} from "@decky/ui";
import { useEffect, useRef, useState } from "react";

type Multiplier = 0 | 2 | 3;
type RefreshHz = 60 | 120;

type Profile = {
  multiplier: Multiplier;
  source_fps_cap: number | null;
  flow_analysis_percent: number;
  target_refresh_hz: RefreshHz;
  auto_display_mode: boolean;
};

type Size = { width: number; height: number };

type Runtime = {
  active: boolean;
  frame_size?: Size | null;
  applied_multiplier: Multiplier;
  reason: string;
  original_frames: number;
  generated_frames: number;
  original_submission_fps: number | null;
  total_submission_fps: number | null;
};

type EngineInfo = {
  ready: boolean;
  active_version: string | null;
  resolutions: string[];
  qairt_ok: boolean;
  launch_token: string;
  npu_preset?: string;
};

type TierRuntime = {
  pid: number;
  preset: string;
  scale: string;
  scale_w: number;
  scale_h: number;
  native_w: number;
  native_h: number;
  native: boolean;
  max_phases: number;
  active_phases: number;
  pair_ema_ms: number;
  guard_cap: string;
};

type Status = {
  stored: boolean;
  profile: Profile;
  runtime: Runtime | null;
  engine: EngineInfo;
  tier_runtime?: TierRuntime | null;
};

type SaveResult = {
  launch_options: string | null;
  launch_changed: boolean;
  launch_warning: string | null;
  engine: EngineInfo;
};

type DetectedGame = { appid: string; name: string; running: boolean };

const readStatus = callable<[appId: number], Status>("game_status");
const saveProfile = callable<
  [
    appId: number,
    multiplier: Multiplier,
    sourceFpsCap: number | null,
    targetRefreshHz: RefreshHz,
    autoDisplayMode: boolean,
    currentLaunchOptions: string | null,
  ],
  SaveResult
>("save_game_profile");

function withTimeout<T>(request: Promise<T>, message: string, ms = 6000): Promise<T> {
  let timer: number | undefined;
  const timeout = new Promise<never>((_, reject) => {
    timer = window.setTimeout(() => reject(new Error(message)), ms);
  });
  return Promise.race([request, timeout]).finally(() => window.clearTimeout(timer));
}

function makeSerial() {
  let tail: Promise<unknown> = Promise.resolve();
  let depth = 0;
  return {
    run<T>(fn: () => Promise<T>): Promise<T> {
      depth++;
      const p = tail.then(fn, fn);
      tail = p.then(
        () => {
          depth--;
        },
        () => {
          depth--;
        }
      );
      return p;
    },
    busy(): boolean {
      return depth > 0;
    },
  };
}

const serial = makeSerial();

function getSteamGlobal(): any {
  return window as any;
}

function isShortcutApp(appId: number): boolean {
  try {
    const ov = getSteamGlobal().appStore?.GetAppOverviewByAppID?.(appId);
    if (ov?.app_type === 1073741824) return true;
  } catch {}
  return appId >= 0x80000000;
}

function readSteamLaunchOptions(appId: number): Promise<string | null> {
  return new Promise((resolve) => {
    let done = false;
    let reg: any = null;
    const finish = (val: string | null) => {
      if (done) return;
      done = true;
      window.setTimeout(() => {
        try {
          reg?.unregister?.();
        } catch {}
      }, 0);
      resolve(val);
    };
    try {
      const sc = getSteamGlobal().SteamClient;
      if (!sc?.Apps?.RegisterForAppDetails) {
        finish(null);
        return;
      }
      const key = isShortcutApp(appId) ? "strShortcutLaunchOptions" : "strLaunchOptions";
      reg = sc.Apps.RegisterForAppDetails(appId, (details: any) => {
        if (typeof details?.[key] === "string") {
          finish(details[key]);
        }
      });
    } catch {
      finish(null);
    }
    window.setTimeout(() => finish(null), 3500);
  });
}

function writeSteamLaunchOptions(appId: number, options: string): boolean {
  try {
    const sc = getSteamGlobal().SteamClient;
    if (isShortcutApp(appId) && sc?.Apps?.SetShortcutLaunchOptions) {
      sc.Apps.SetShortcutLaunchOptions(appId, options);
      return true;
    }
    if (sc?.Apps?.SetAppLaunchOptions) {
      sc.Apps.SetAppLaunchOptions(appId, options);
      return true;
    }
  } catch {}
  return false;
}

function detectRunningGame(): DetectedGame | null {
  const app = Router.MainRunningApp;
  if (!app?.appid) return null;
  return { appid: String(app.appid), name: app.display_name || `游戏 ${app.appid}`, running: true };
}

function detectLibraryPageGame(): DetectedGame | null {
  try {
    const g = getSteamGlobal();
    const path: string =
      (Router as any).WindowStore?.GamepadUIMainWindowInstance?.m_history?.location?.pathname ?? "";
    const match = path.match(/^\/library\/app\/(\d+)/);
    if (!match) return null;
    const id = match[1];
    const ov = g.appStore?.GetAppOverviewByAppID?.(Number(id));
    return { appid: id, name: ov?.display_name || `游戏 ${id}`, running: false };
  } catch {
    return null;
  }
}

function currentGame(): DetectedGame | null {
  return detectRunningGame() ?? detectLibraryPageGame();
}

function hasLauncherToken(opts: string | null): boolean {
  if (!opts) return false;
  return opts.includes(".local/share/odin3-fg/bin/odin3-fg-launch");
}

const MODE_CARDS: { value: Multiplier; title: string; subtitle: string }[] = [
  { value: 0, title: "关闭（原生直通）", subtitle: "按游戏原设置运行 · 自动移除启动项" },
  { value: 2, title: "2× 分档混合补帧", subtitle: "原生无损送屏 · 60/80 → 120 Hz 自适应插帧" },
  { value: 3, title: "3× 分档混合补帧", subtitle: "原生无损送屏 · 40/60/80 → 120 Hz 动态单/双相位" },
];

const SECTION_HEAD: Record<string, string | number> = {
  fontSize: "12px",
  fontWeight: 600,
  opacity: 0.85,
  marginBottom: "4px",
};

function StatusCard({
  game,
  status,
  launchOpts,
}: {
  game: DetectedGame;
  status: Status;
  launchOpts: string | null;
}) {
  const { runtime, profile, engine } = status;
  const mounted = hasLauncherToken(launchOpts);

  let dot = "#9ca3af";
  let headline = "当前未开启补帧（原生运行）";
  if (runtime?.active) {
    dot = "#4ade80";
    headline = `纯神经 ${runtime.applied_multiplier}× 补帧正在运行`;
  } else if (runtime && !runtime.active) {
    dot = "#facc15";
    headline =
      runtime.reason === "synchronizing-frame-cadence" || runtime.generated_frames > 0
        ? "交换链已连接，正在同步帧率节奏…"
        : `已注入，等待游戏匹配交换链 (${runtime.reason})`;
  } else if (profile.multiplier > 0) {
    dot = mounted ? "#60a5fa" : "#fb923c";
    headline = mounted
      ? `已启用 ${profile.multiplier}× 纯神经补帧（下次启动生效）`
      : `已选 ${profile.multiplier}×（正在同步启动项…）`;
  }

  return (
    <div
      style={{
        borderRadius: "6px",
        padding: "10px 12px",
        background: "rgba(255,255,255,0.06)",
        fontSize: "12px",
        lineHeight: "1.5",
      }}
      aria-live="polite"
    >
      <div style={{ fontSize: "14px", fontWeight: 600, marginBottom: "4px" }}>
        {game.running ? `正在运行：${game.name}` : game.name}
      </div>
      <div style={{ fontSize: "13px", fontWeight: 600 }}>
        <span style={{ color: dot }}>●</span> {headline}
      </div>
      {runtime?.frame_size && (
        <div style={{ marginTop: "2px" }}>
          交换链分辨率：{runtime.frame_size.width}×{runtime.frame_size.height}
          {status.tier_runtime
            ? ` · 档位 ${status.tier_runtime.preset.toUpperCase()} → ${
                status.tier_runtime.native ? `原生 ${status.tier_runtime.scale}` : `${status.tier_runtime.scale} 混合超分`
              } (${status.tier_runtime.pair_ema_ms.toFixed(1)} ms)`
            : engine.npu_preset
              ? ` · 当前 NPU 档位 ${engine.npu_preset.toUpperCase()}`
              : ""}
        </div>
      )}
      {!runtime?.frame_size && engine.npu_preset && (
        <div style={{ marginTop: "2px", opacity: 0.85 }}>
          联动 NPU 功耗档位：{engine.npu_preset.toUpperCase()}（随档自动匹配 360p–1080p 补帧管线）
        </div>
      )}
      {runtime?.original_submission_fps != null && runtime.total_submission_fps != null && (
        <div style={{ color: "#86efac", fontWeight: 600, marginTop: "2px" }}>
          实时帧率：原帧 {runtime.original_submission_fps.toFixed(1)} → 总提呈{" "}
          {runtime.total_submission_fps.toFixed(1)} FPS
        </div>
      )}
      {runtime?.active && runtime.applied_multiplier !== profile.multiplier && (
        <div style={{ color: "#fde047", marginTop: "4px" }}>
          → 已切换新档位，重启游戏后生效。
        </div>
      )}
      <div style={{ opacity: 0.75, marginTop: "4px" }}>
        引擎：{engine.ready ? `OmniShift HTP (${engine.resolutions.join("/")}) · 120Hz已校准` : "运行包未就绪"}
        {launchOpts !== null && ` · 启动项${mounted ? "已挂载" : "原生"}`}
      </div>
    </div>
  );
}

function Content() {
  const visible = useQuickAccessVisible();
  const [game, setGame] = useState<DetectedGame | null>(currentGame);
  const [status, setStatus] = useState<Status | null>(null);
  const [statusAppId, setStatusAppId] = useState<number>(0);
  const [launchOpts, setLaunchOpts] = useState<string | null>(null);
  const [busy, setBusy] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const checkedLaunchAppRef = useRef<number>(0);

  const appId = game ? Number(game.appid) : 0;

  useEffect(() => {
    if (!visible) return;
    checkedLaunchAppRef.current = 0;
    let stopped = false;
    let timer: number | undefined;

    async function poll() {
      const g = currentGame();
      if (!stopped) setGame(g);
      const id = g ? Number(g.appid) : 0;
      if (!id || serial.busy()) {
        if (!stopped) timer = window.setTimeout(poll, 1500);
        return;
      }
      try {
        const firstForApp = checkedLaunchAppRef.current !== id;
        const [st, lo] = await Promise.all([
          withTimeout(readStatus(id), "Decky 后端响应超时"),
          firstForApp ? readSteamLaunchOptions(id) : Promise.resolve(undefined),
        ]);
        if (stopped || serial.busy()) {
          if (!stopped) timer = window.setTimeout(poll, 1500);
          return;
        }
        if (firstForApp) {
          checkedLaunchAppRef.current = id;
        }
        setStatus(st);
        setStatusAppId(id);
        if (lo !== undefined) {
          setLaunchOpts(lo);
          // Self-heal Steam Launch Options if profile > 0 but token missing
          if (lo !== null && st.profile.multiplier > 0 && !hasLauncherToken(lo)) {
            void applyChange(
              id,
              st.profile.multiplier,
              st.profile.source_fps_cap,
              st.profile.target_refresh_hz || 120,
              st.profile.auto_display_mode ?? true,
              true
            );
          }
        }
        setError(null);
      } catch (err) {
        if (!stopped) setError(String(err));
      }
      if (!stopped) {
        timer = window.setTimeout(poll, 1500);
      }
    }

    void poll();
    return () => {
      stopped = true;
      window.clearTimeout(timer);
    };
  }, [visible]);

  async function applyChange(
    targetAppId: number,
    nextMultiplier: Multiplier,
    nextCap: number | null,
    nextHz: RefreshHz,
    nextAutoDisp: boolean,
    silent = false
  ) {
    // Optimistic UI update immediately so button highlight never flickers!
    setStatus((prev) =>
      prev && statusAppId === targetAppId
        ? {
            ...prev,
            stored: true,
            profile: {
              ...prev.profile,
              multiplier: nextMultiplier,
              source_fps_cap: nextCap,
              target_refresh_hz: nextHz,
              auto_display_mode: nextAutoDisp,
            },
          }
        : prev
    );
    setBusy(true);
    try {
      await serial.run(async () => {
        const currentLo = await readSteamLaunchOptions(targetAppId);
        const res = await withTimeout(
          saveProfile(targetAppId, nextMultiplier, nextCap, nextHz, nextAutoDisp, currentLo),
          "保存补帧配置超时"
        );
        if (res.launch_changed && res.launch_options !== null) {
          writeSteamLaunchOptions(targetAppId, res.launch_options);
          setLaunchOpts(res.launch_options);
        } else if (res.launch_options !== null) {
          setLaunchOpts(res.launch_options);
        }
        const refreshed = await withTimeout(readStatus(targetAppId), "刷新状态超时");
        setStatus(refreshed);
        setStatusAppId(targetAppId);
        if (!silent) {
          if (res.launch_warning) {
            toaster.toast({ title: "Odin3 NPU 补帧", body: res.launch_warning });
          } else if (nextMultiplier === 0) {
            toaster.toast({ title: "Odin3 NPU 补帧", body: "已关闭补帧并恢复原生启动项" });
          } else {
            toaster.toast({
              title: `已启用 ${nextMultiplier}× 纯神经补帧`,
              body: `目标 ${nextHz} Hz · 原帧上限 ${nextCap ?? "不限"} FPS`,
            });
          }
        }
      });
    } catch (err) {
      toaster.toast({ title: "保存失败", body: String(err) });
    } finally {
      setBusy(false);
    }
  }

  if (!game || !appId) {
    return (
      <PanelSection title="Odin3 分档混合补帧">
        <PanelSectionRow>
          <div style={{ fontSize: "12px", opacity: 0.85, lineHeight: "1.5" }}>
            请先在 Steam 库中点开任意游戏的详情页，或直接启动游戏后再打开本面板（按游戏独立保存档位）。
          </div>
        </PanelSectionRow>
      </PanelSection>
    );
  }

  if (!status || statusAppId !== appId) {
    return (
      <PanelSection title="Odin3 分档混合补帧">
        <PanelSectionRow>
          <div style={{ fontSize: "12px" }}>{error ?? `正在读取 ${game.name} 的配置…`}</div>
        </PanelSectionRow>
      </PanelSection>
    );
  }

  const profile = status.profile;
  const mult = profile.multiplier;
  const hz: RefreshHz = profile.target_refresh_hz === 60 ? 60 : 120;
  const cap = profile.source_fps_cap;
  const autoDisp = profile.auto_display_mode ?? true;
  const suggestedCap = mult > 0 ? Math.round(hz / mult) : 60;

  function onPickMultiplier(nextMult: Multiplier) {
    if (nextMult === mult && (nextMult === 0 || hasLauncherToken(launchOpts))) return;
    // clear legacy 60/40/30/20 auto-caps when enabling FG so 80+ FPS games retain 100% native frames!
    const nextCap = (nextMult > 0 && (mult === 0 || cap === 60 || cap === 40 || cap === 30 || cap === 20)) ? null : cap;
    void applyChange(appId, nextMult, nextCap, hz, autoDisp);
  }

  function onPickHz(nextHz: RefreshHz) {
    if (nextHz === hz) return;
    void applyChange(appId, mult, cap, nextHz, autoDisp);
  }

  function onPickCap(nextCap: number | null) {
    if (nextCap === cap) return;
    void applyChange(appId, mult, nextCap, hz, autoDisp);
  }

  void suggestedCap;
  const capPresets: { label: string; value: number | null }[] = [
    { label: "不限（原生无损）", value: null },
    { label: "60", value: 60 },
    { label: "45", value: 45 },
    { label: "40", value: 40 },
    { label: "30", value: 30 },
  ];

  return (
    <PanelSection title="Odin3 分档混合补帧">
      <PanelSectionRow>
        <StatusCard game={game} status={status} launchOpts={launchOpts} />
      </PanelSectionRow>

      <PanelSectionRow>
        <div style={SECTION_HEAD}>补帧档位（点击即生效）</div>
      </PanelSectionRow>
      <PanelSectionRow>
        <Focusable style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
          {MODE_CARDS.map((card) => {
            const active = mult === card.value;
            return (
              <DialogButton
                key={card.value}
                disabled={busy}
                onClick={() => onPickMultiplier(card.value)}
                style={{
                  minWidth: 0,
                  padding: "8px 12px",
                  textAlign: "left",
                  ...(active ? { background: "#1a9fff", color: "#ffffff" } : {}),
                }}
              >
                <div style={{ fontSize: "14px", fontWeight: 600 }}>
                  {active ? `✓ ${card.title}` : card.title}
                </div>
                <div style={{ fontSize: "11px", opacity: active ? 0.95 : 0.72, marginTop: "2px" }}>
                  {card.subtitle}
                </div>
              </DialogButton>
            );
          })}
        </Focusable>
      </PanelSectionRow>

      {mult > 0 && (
        <>
          <PanelSectionRow>
            <div style={SECTION_HEAD}>目标屏幕刷新率</div>
          </PanelSectionRow>
          <PanelSectionRow>
            <Focusable style={{ display: "flex", gap: "6px" }}>
              {([120, 60] as RefreshHz[]).map((optionHz) => {
                const active = hz === optionHz;
                return (
                  <DialogButton
                    key={optionHz}
                    disabled={busy}
                    onClick={() => onPickHz(optionHz)}
                    style={{
                      minWidth: 0,
                      flex: 1,
                      padding: "7px 8px",
                      fontSize: "13px",
                      fontWeight: 600,
                      ...(active ? { background: "#1a9fff", color: "#ffffff" } : {}),
                    }}
                  >
                    {active ? `✓ ${optionHz} Hz` : `${optionHz} Hz`}
                  </DialogButton>
                );
              })}
            </Focusable>
          </PanelSectionRow>

          <PanelSectionRow>
            <div style={SECTION_HEAD}>
              原帧率上限（当前：{cap !== null ? `${cap} FPS` : "游戏自身限帧"}）
            </div>
          </PanelSectionRow>
          <PanelSectionRow>
            <Focusable style={{ display: "flex", flexWrap: "wrap", gap: "6px" }}>
              {capPresets.map((preset) => {
                const active = cap === preset.value;
                return (
                  <DialogButton
                    key={String(preset.value)}
                    disabled={busy}
                    onClick={() => onPickCap(preset.value)}
                    style={{
                      minWidth: "30%",
                      flex: "1 1 30%",
                      padding: "6px 6px",
                      fontSize: "12px",
                      fontWeight: 600,
                      ...(active ? { background: "#1a9fff", color: "#ffffff" } : {}),
                    }}
                  >
                    {active ? `✓ ${preset.label}` : preset.label}
                  </DialogButton>
                );
              })}
            </Focusable>
          </PanelSectionRow>

          <PanelSectionRow>
            <ToggleField
              label="自动切换刷新率与 SDR"
              description="启动时自动设为目标刷新率与 SDR 零拷贝直呈，退出游戏自动恢复。"
              checked={autoDisp}
              disabled={busy}
              onChange={(checked: boolean) => void applyChange(appId, mult, cap, hz, checked)}
            />
          </PanelSectionRow>
        </>
      )}

      <PanelSectionRow>
        <div style={{ fontSize: "11px", opacity: 0.72, lineHeight: "1.45" }}>
          联动 Odin3 NPU Power 全档位（Eco–Max 自适应 360p–1080p 混合加速，原生帧零损直出）。
        </div>
      </PanelSectionRow>
    </PanelSection>
  );
}

export default definePlugin(() => ({
  name: "Odin3 FG NPU",
  titleView: <div className={staticClasses.Title}>Odin3 NPU 补帧</div>,
  content: <Content />,
  icon: <span aria-hidden="true" style={{ fontWeight: 700, fontSize: "13px" }}>NPU</span>,
}));
