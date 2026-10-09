<script setup lang="ts">
import {
  Loader2,
  Pencil,
  Play,
  Plus,
  Server,
  Trash2,
  Zap,
} from "lucide-vue-next";
import { toast } from "vue-sonner";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Empty,
  EmptyContent,
  EmptyDescription,
  EmptyHeader,
  EmptyMedia,
  EmptyTitle,
} from "@/components/ui/empty";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { presetLabel, type EndpointPreset } from "@/composables/usePresets";
import { percent } from "@/lib/format";
import { cn } from "@/lib/utils";

const { presets, storageError, addPreset, updatePreset, removePreset } =
  usePresets();
const { bank, error: bankError, load } = useBank();
const {
  runStates,
  batchRunning,
  queuedIds,
  isRunning,
  runPreset,
  runBatch,
  clearRun,
} = useApiTest();
const selectedId = ref<string | null>(null);
const selected = computed(
  () => presets.value.find((preset) => preset.id === selectedId.value) ?? null,
);
const dialogOpen = ref(false);
const editing = ref<EndpointPreset | null>(null);
const deleting = ref<EndpointPreset | null>(null);
const deleteOpen = ref(false);
const proxyBaseURL = useRuntimeConfig().public.proxyUrl as string;
// 暂时隐藏终端入口，保留实现以便后续恢复。
const terminalEnabled = false;
const proxyOpen = ref(false);
const {
  allowed: proxyAllowed,
  allow: allowProxy,
  revoke: revokeProxy,
} = useProxyConsent(proxyBaseURL);
const pendingTest = ref<{ targets: EndpointPreset[]; batch: boolean } | null>(
  null,
);
const terminalOpen = ref(false);
const terminalPreset = ref<EndpointPreset | null>(null);
const { sessions: terminalSessions, clear: clearTerminal } = useTerminalTest();

function requestTests(targets: EndpointPreset[], batch: boolean) {
  if (!targets.length || proxyOpen.value) return;
  selectedId.value = targets[0]!.id;
  if (proxyBaseURL && !proxyAllowed.value) {
    pendingTest.value = {
      targets: targets.map((preset) => ({ ...preset })),
      batch,
    };
    proxyOpen.value = true;
    return;
  }
  void executeTests(
    targets,
    batch,
    proxyAllowed.value ? proxyBaseURL : undefined,
  );
}
function isNetworkFailure(id: string) {
  const state = runStates.value[id];
  return (
    state?.status === "failed" &&
    state.errors.some((message) =>
      /failed to fetch|networkerror|network error|load failed|cors/i.test(
        message,
      ),
    )
  );
}
function consentProxy() {
  const pending = pendingTest.value;
  pendingTest.value = null;
  proxyOpen.value = false;
  if (!pending || !proxyBaseURL) return;
  allowProxy();
  void executeTests(pending.targets, pending.batch, proxyBaseURL);
}
function declineProxy() {
  const pending = pendingTest.value;
  pendingTest.value = null;
  proxyOpen.value = false;
  if (pending) void executeTests(pending.targets, pending.batch);
}
function openTerminal(preset: EndpointPreset) {
  selectedId.value = preset.id;
  terminalPreset.value = preset;
  terminalOpen.value = true;
}

watch(
  () => presets.value.map((preset) => preset.id),
  (ids) => {
    if (!selectedId.value || !ids.includes(selectedId.value))
      selectedId.value = ids[0] ?? null;
  },
  { immediate: true },
);
onMounted(load);

function locked(id: string) {
  return proxyOpen.value || batchRunning.value || isRunning(id);
}
function status(id: string) {
  if (queuedIds.value.includes(id)) return "排队中";
  const state = runStates.value[id];
  if (state?.status === "running") return "测试中";
  if (state?.result || (terminalEnabled && terminalSessions.value[id]?.result))
    return "已完成";
  return state?.status === "failed" ? "失败" : "未测试";
}
function configure(preset: EndpointPreset | null = null) {
  editing.value = preset;
  dialogOpen.value = true;
}
function save(input: Omit<EndpointPreset, "id">) {
  if (editing.value) {
    if (locked(editing.value.id)) return;
    updatePreset(editing.value.id, input);
    clearRun(editing.value.id);
    clearTerminal(editing.value.id);
    selectedId.value = editing.value.id;
  } else {
    selectedId.value = addPreset(input).id;
  }
  toast.success("服务商配置已保存", { id: "provider-saved", duration: 1800 });
}
function confirmDelete() {
  if (!deleting.value || locked(deleting.value.id)) return;
  clearRun(deleting.value.id);
  clearTerminal(deleting.value.id);
  removePreset(deleting.value.id);
  deleting.value = null;
  deleteOpen.value = false;
  toast.success("服务商已删除");
}
function test(preset: EndpointPreset) {
  if (!bank.value || locked(preset.id)) return;
  requestTests([preset], false);
}
function testAll() {
  if (!bank.value || batchRunning.value || proxyOpen.value) return;
  requestTests(
    presets.value.filter((preset) => !isRunning(preset.id)),
    true,
  );
}
async function executeTests(
  targets: EndpointPreset[],
  batch: boolean,
  proxyURL?: string,
) {
  try {
    targets.forEach((preset) => clearTerminal(preset.id));
    if (batch) {
      await runBatch(targets, proxyURL);
      const success = targets.filter(
        (preset) => runStates.value[preset.id]?.status === "success",
      ).length;
      toast.success(
        `批量测试完成：成功 ${success} 个，失败 ${targets.length - success} 个`,
      );
    } else {
      const preset = targets[0]!;
      const state = await runPreset(preset, proxyURL);
      if (state.status === "failed") {
        toast.error(`「${presetLabel(preset)}」测试失败，请查看详情`);
      }
    }
  } catch (error) {
    toast.error(error instanceof Error ? error.message : "无法开始测试");
  }
}
</script>

<template>
  <div
    class="mt-3 flex min-h-0 flex-1 flex-col gap-3 overflow-hidden rounded-md bg-card p-3 lg:mt-4 lg:p-4"
  >
    <div
      class="flex shrink-0 flex-wrap items-center justify-between gap-3 px-1"
    >
      <div class="flex items-baseline gap-2">
        <h1 class="text-sm font-semibold">自动测试</h1>
        <span class="text-xs text-muted-foreground"
          >{{ presets.length }} 个服务商 ·
          {{ proxyAllowed ? "服务端代理已授权" : "浏览器直连" }}</span
        >
      </div>
      <div class="flex items-center gap-2">
        <Button
          v-if="proxyAllowed"
          :disabled="
            batchRunning || presets.some((preset) => isRunning(preset.id))
          "
          title="测试结束后可撤销代理授权"
          size="sm"
          variant="ghost"
          @click="revokeProxy"
          >撤销代理授权</Button
        >
        <Button size="sm" variant="outline" @click="configure()"
          ><Plus data-icon="inline-start" />添加服务商</Button
        >
        <Button
          size="sm"
          :disabled="
            !bank ||
            !presets.length ||
            batchRunning ||
            proxyOpen ||
            presets.every((p) => isRunning(p.id))
          "
          @click="testAll"
        >
          <Loader2
            v-if="batchRunning"
            class="animate-spin"
            data-icon="inline-start"
          />
          <Zap v-else data-icon="inline-start" />
          {{ batchRunning ? "批量测试中…" : "一键测全部" }}
        </Button>
      </div>
    </div>
    <Alert v-if="storageError || bankError" variant="destructive">
      <AlertDescription>{{ storageError || bankError }}</AlertDescription>
    </Alert>
    <div
      class="grid min-h-0 flex-1 gap-6 lg:grid-cols-[minmax(0,38.2fr)_minmax(0,61.8fr)] lg:grid-rows-[minmax(0,1fr)]"
    >
      <section
        aria-label="服务商列表"
        class="flex min-h-0 flex-col gap-3 overflow-y-auto"
      >
        <Empty v-if="!presets.length" class="min-h-60 rounded-md border">
          <EmptyHeader>
            <EmptyMedia variant="icon"><Server /></EmptyMedia>
            <EmptyTitle>添加服务商</EmptyTitle>
            <EmptyDescription
              >配置名称、Endpoint、API Key 和模型
              ID。可以添加多组配置，分别测试或一键测试全部。</EmptyDescription
            >
          </EmptyHeader>
          <EmptyContent
            ><Button size="sm" @click="configure()"
              ><Plus data-icon="inline-start" />添加第一个服务商</Button
            ></EmptyContent
          >
        </Empty>
        <article
          v-for="preset in presets"
          :key="preset.id"
          :class="
            cn(
              'shrink-0 rounded-md border p-3.5 transition-colors',
              selectedId === preset.id && 'border-primary bg-muted/40',
            )
          "
        >
          <button
            type="button"
            class="flex w-full flex-col gap-2 rounded-sm text-left focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
            :aria-pressed="selectedId === preset.id"
            @click="selectedId = preset.id"
          >
            <span class="flex w-full items-center justify-between gap-2">
              <strong class="min-w-0 break-all text-sm">{{
                presetLabel(preset)
              }}</strong>
              <Badge
                :variant="
                  status(preset.id) === '失败' ? 'destructive' : 'secondary'
                "
                class="shrink-0 font-normal"
              >
                <Loader2
                  v-if="isRunning(preset.id)"
                  class="mr-1 size-3 animate-spin"
                />{{ status(preset.id) }}
              </Badge>
            </span>
            <span
              class="max-w-full truncate text-xs text-muted-foreground"
              :title="preset.baseUrl"
              >{{ preset.baseUrl }}</span
            >
            <span
              class="flex flex-wrap items-center gap-2 text-xs text-muted-foreground"
            >
              <span class="break-all font-mono">{{ preset.model }}</span>
              <Badge variant="outline" class="font-normal">{{
                preset.apiType === "responses" ? "Responses" : "Chat"
              }}</Badge>
            </span>
            <span
              v-if="
                runStates[preset.id]?.result ||
                (terminalEnabled && terminalSessions[preset.id]?.result)
              "
              class="flex flex-wrap items-baseline gap-2 text-sm"
            >
              <strong class="uppercase">{{
                ((terminalEnabled && terminalSessions[preset.id]?.result) ||
                  runStates[preset.id]?.result)!.prediction_name
              }}</strong>
              <span class="tnum">{{
                percent(
                  ((terminalEnabled && terminalSessions[preset.id]?.result) ||
                    runStates[preset.id]?.result)!.probability,
                )
              }}</span>
              <span
                v-if="isRunning(preset.id)"
                class="text-xs text-muted-foreground"
                >更新中</span
              >
            </span>
          </button>
          <div class="mt-3 flex items-center justify-between gap-2">
            <Button
              size="sm"
              variant="outline"
              :disabled="!bank || locked(preset.id)"
              :aria-label="`测试 ${presetLabel(preset)}`"
              @click="test(preset)"
            >
              <Play data-icon="inline-start" />{{
                isRunning(preset.id)
                  ? "测试中…"
                  : runStates[preset.id]
                    ? "重新测试"
                    : "开始测试"
              }}
            </Button>
            <div class="flex items-center gap-1">
              <Button
                size="icon"
                variant="ghost"
                :disabled="locked(preset.id)"
                :aria-label="`编辑 ${presetLabel(preset)}`"
                @click="configure(preset)"
                ><Pencil
              /></Button>
              <Button
                size="icon"
                variant="ghost"
                :disabled="locked(preset.id)"
                :aria-label="`删除 ${presetLabel(preset)}`"
                @click="
                  deleting = preset;
                  deleteOpen = true;
                "
                ><Trash2
              /></Button>
            </div>
          </div>
        </article>
        <p class="px-1 text-xs leading-relaxed text-muted-foreground">
          每个服务商最多测试 3 题；归因概率达到 99%
          即成功检验，不再请求后续题目。 批量并发 2 个服务商；测试会消耗所填 API
          的额度。直连需要服务商允许跨域（CORS），也可在测试前授权使用代理。
        </p>
      </section>
      <section aria-label="服务商测试详情" class="min-h-0 overflow-y-auto">
        <ProviderTestDetail
          v-if="selected"
          :preset="selected"
          :run="runStates[selected.id]"
          :terminal="
            terminalEnabled ? terminalSessions[selected.id] : undefined
          "
          :terminal-available="terminalEnabled"
          :proxy-available="
            !!proxyBaseURL && !proxyAllowed && isNetworkFailure(selected.id)
          "
          :queued="queuedIds.includes(selected.id)"
          :disabled="!bank || locked(selected.id)"
          @test="test(selected)"
          @terminal="openTerminal(selected)"
          @proxy="test(selected)"
        />
        <Empty v-else class="min-h-80 rounded-md border lg:h-full">
          <EmptyHeader>
            <EmptyMedia variant="icon"><Server /></EmptyMedia>
            <EmptyTitle>服务商测试详情</EmptyTitle>
            <EmptyDescription
              >添加并选择左侧服务商后，在这里查看挑战、模型回答和归因结果。</EmptyDescription
            >
          </EmptyHeader>
        </Empty>
      </section>
    </div>
    <AlertDialog v-model:open="proxyOpen">
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>是否通过 Cloudflare 代理测试？</AlertDialogTitle>
          <AlertDialogDescription>
            即将{{ pendingTest?.batch ? "批量测试" : "测试" }}
            {{ pendingTest?.targets.length || 0 }} 个服务商。 若同意，你的 API
            Key、模型 ID 和挑战提示词将发送到
            {{ proxyBaseURL }}，再由该服务器请求服务商。
            授权会在此浏览器记住，后续单个和批量测试均直接使用此代理，不再询问；可在测试结束后撤销授权。
            这不再是纯浏览器直连；请仅在信任代理运营方时继续。
          </AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel @click="declineProxy"
            >仅本次直连</AlertDialogCancel
          >
          <AlertDialogAction @click="consentProxy"
            >同意并通过代理测试</AlertDialogAction
          >
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
    <TerminalTestDialog
      v-if="terminalEnabled"
      v-model:open="terminalOpen"
      :preset="terminalPreset"
    />
    <PresetFormDialog
      v-model:open="dialogOpen"
      :preset="editing"
      @save="save"
    />
    <AlertDialog v-model:open="deleteOpen">
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>删除服务商</AlertDialogTitle>
          <AlertDialogDescription
            >确定删除「{{
              deleting ? presetLabel(deleting) : ""
            }}」及其测试结果吗？此操作不可撤销。</AlertDialogDescription
          >
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel>取消</AlertDialogCancel>
          <AlertDialogAction @click="confirmDelete">删除</AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  </div>
</template>
