// Endpoint 预设的本地持久化（localStorage，带版本 schema，参考 client-localstorage-schema 规则）

export type ApiType = "chat" | "responses";

export interface EndpointPreset {
  id: string;
  name: string;
  baseUrl: string;
  apiKey: string;
  model: string;
  apiType: ApiType;
  /** null 表示不设置，由服务端默认 */
  temperature: number | null;
}

interface PresetStorage {
  version: 1;
  presets: EndpointPreset[];
}

const STORAGE_KEY = "modeltrace.presets.v1";

/** 展示名：未填名称时回退为域名 */
export function presetLabel(
  preset: Pick<EndpointPreset, "name" | "baseUrl">,
): string {
  const name = preset.name.trim();
  if (name) return name;
  try {
    return new URL(preset.baseUrl).hostname;
  } catch {
    return preset.baseUrl;
  }
}

function readStorage(): EndpointPreset[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw) as Partial<PresetStorage>;
    if (parsed.version !== 1 || !Array.isArray(parsed.presets)) return [];
    const ids = new Set<string>();
    return parsed.presets
      .filter((item): item is EndpointPreset => {
        if (
          !item ||
          typeof item.id !== "string" ||
          !item.id ||
          ids.has(item.id)
        )
          return false;
        if (
          ![item.baseUrl, item.apiKey, item.model].every(
            (value) => typeof value === "string" && value.trim(),
          )
        )
          return false;
        if (typeof item.name !== "string") return false;
        if (item.apiType !== "chat" && item.apiType !== "responses")
          return false;
        ids.add(item.id);
        return true;
      })
      .map((item) => ({
        ...item,
        // 兼容此前名称可选的预设，不丢弃已有配置。
        name: presetLabel(item),
        temperature:
          typeof item.temperature === "number" &&
          Number.isFinite(item.temperature)
            ? item.temperature
            : null,
      }));
  } catch {
    return [];
  }
}

export function usePresets() {
  // 惰性初始化（rerender-lazy-state-init）：ssr: false，setup 只在浏览器执行
  const presets = useState<EndpointPreset[]>("modeltrace:presets", () =>
    readStorage(),
  );

  const storageError = useState<string | null>(
    "modeltrace:presets:storage-error",
    () => null,
  );

  function persist() {
    try {
      const payload: PresetStorage = { version: 1, presets: presets.value };
      localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
      storageError.value = null;
    } catch {
      storageError.value =
        "浏览器存储不可用，配置仅在本次会话中保留，刷新后会丢失。";
    }
  }

  function addPreset(input: Omit<EndpointPreset, "id">): EndpointPreset {
    const preset: EndpointPreset = { ...input, id: crypto.randomUUID() };
    presets.value = [...presets.value, preset];
    persist();
    return preset;
  }

  function updatePreset(id: string, input: Omit<EndpointPreset, "id">) {
    presets.value = presets.value.map((item) =>
      item.id === id ? { ...input, id } : item,
    );
    persist();
  }

  function removePreset(id: string) {
    presets.value = presets.value.filter((item) => item.id !== id);
    persist();
  }

  return { presets, storageError, addPreset, updatePreset, removePreset };
}
