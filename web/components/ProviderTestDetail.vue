<script setup lang="ts">
import { Loader2, Play, ScanSearch, Terminal } from "lucide-vue-next";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Empty,
  EmptyDescription,
  EmptyHeader,
  EmptyMedia,
  EmptyTitle,
} from "@/components/ui/empty";
import { Progress } from "@/components/ui/progress";
import { Separator } from "@/components/ui/separator";
import { presetLabel, type EndpointPreset } from "@/composables/usePresets";
import type { PresetRunState, StepState } from "@/composables/useApiTest";
import type { TerminalSession } from "@/composables/useTerminalTest";

defineProps<{
  preset: EndpointPreset;
  run?: PresetRunState;
  terminal?: TerminalSession;
  terminalAvailable: boolean;
  proxyAvailable: boolean;
  queued: boolean;
  disabled: boolean;
}>();
defineEmits<{ test: []; terminal: []; proxy: [] }>();
const labels: Record<StepState, string> = {
  pending: "等待",
  working: "请求中",
  done: "有效",
  invalid: "数字不足",
  error: "请求失败",
  skipped: "未调用",
};
</script>

<template>
  <div class="flex flex-col gap-4">
    <section
      v-if="terminal?.result || run?.result"
      aria-label="归因结果"
      class="flex flex-col gap-3"
    >
      <template v-if="terminal?.result">
        <p class="px-1 text-xs text-muted-foreground">
          终端 curl 回答 · 有效 {{ terminal.result.used_outputs }}/3
        </p>
        <ResultPanel :result="terminal.result" />
      </template>
      <template v-else-if="run?.result">
        <p
          v-if="run.status === 'running'"
          class="px-1 text-xs text-muted-foreground"
        >
          已有初步结果，概率达到 99% 即结束，否则继续测试剩余题目。
        </p>
        <ResultPanel :result="run.result" />
      </template>
    </section>
    <Separator v-if="terminal?.result || run?.result" />
    <div
      aria-label="服务商信息"
      class="flex flex-wrap items-start justify-between gap-3 rounded-md border p-3.5"
    >
      <div class="flex min-w-0 flex-col gap-1.5">
        <h2 class="break-all text-sm font-semibold">
          {{ presetLabel(preset) }}
        </h2>
        <p class="break-all text-xs text-muted-foreground">
          {{ preset.baseUrl }}
        </p>
        <p class="break-all text-xs text-muted-foreground">
          模型 ID：{{ preset.model }} ·
          {{
            preset.apiType === "responses"
              ? "Responses API"
              : "Chat Completions"
          }}
          · 密钥已配置
          <span v-if="run">
            ·
            {{ run.transport === "proxy" ? "服务端代理" : "浏览器直连" }}</span
          >
        </p>
      </div>
      <div class="flex flex-wrap items-center gap-2">
        <Button
          v-if="proxyAvailable"
          :disabled="disabled"
          size="sm"
          variant="outline"
          @click="$emit('proxy')"
          >通过代理测试</Button
        >
        <Button
          v-if="terminalAvailable"
          size="sm"
          variant="outline"
          @click="$emit('terminal')"
        >
          <Terminal data-icon="inline-start" />终端方式
        </Button>
        <Button size="sm" :disabled="disabled" @click="$emit('test')">
          <Loader2
            v-if="run?.status === 'running'"
            class="animate-spin"
            data-icon="inline-start"
          />
          <Play v-else data-icon="inline-start" />
          {{
            queued
              ? "排队中"
              : run?.status === "running"
                ? "测试中…"
                : run
                  ? "重新测试"
                  : "开始测试"
          }}
        </Button>
      </div>
    </div>

    <div
      v-if="run || queued"
      aria-label="调用进度"
      class="flex flex-col gap-2 px-1"
    >
      <p class="text-xs text-muted-foreground" role="status" aria-live="polite">
        {{ queued ? "已加入批量测试队列，等待空闲任务位…" : run?.message }}
      </p>
      <div v-if="run" class="flex items-center gap-3">
        <Progress
          :model-value="
            run.finishedAt
              ? 100
              : (run.steps.filter(
                  (step) => step !== 'pending' && step !== 'working',
                ).length /
                  run.challenges.length) *
                100
          "
          class="h-1.5 flex-1"
        />
        <span class="tnum shrink-0 text-xs text-muted-foreground"
          >有效回答 {{ run.validCount }}/3</span
        >
      </div>
      <p v-if="run?.finishedAt" class="tnum text-xs text-muted-foreground">
        本轮耗时 {{ ((run.finishedAt - run.startedAt) / 1000).toFixed(1) }} 秒
      </p>
    </div>
    <Alert
      v-if="run?.status === 'failed' && !queued && !terminal?.result"
      variant="destructive"
    >
      <AlertDescription
        >{{ run.message }} 请检查 Endpoint、密钥、模型 ID
        和跨域配置；若浏览器跨域被阻止，{{
          proxyAvailable
            ? "可在确认信任代理后选择「通过代理测试」。"
            : "请检查服务商的 CORS 配置。"
        }}</AlertDescription
      >
    </Alert>

    <Empty v-if="!run && !terminal?.result" class="min-h-60 rounded-md border">
      <EmptyHeader>
        <EmptyMedia variant="icon"><ScanSearch /></EmptyMedia>
        <EmptyTitle>{{ queued ? "等待开始" : "准备开始测试" }}</EmptyTitle>
        <EmptyDescription
          >使用 AI SDK 调用此服务商，最多测试 3 题。
          第一份有效回答返回后即可展示初步归因；概率达到 99%
          即成功检验并停止后续调用。</EmptyDescription
        >
      </EmptyHeader>
    </Empty>

    <section v-if="run" aria-label="挑战与模型回答" class="flex flex-col gap-3">
      <h3 class="px-1 text-sm font-semibold">
        {{ terminal?.result ? "浏览器直连尝试" : "挑战与模型回答" }}
      </h3>
      <details
        v-for="(challenge, index) in run.challenges"
        :key="challenge.id"
        class="rounded-md border p-3.5"
        :open="run.steps[index] === 'working' || run.steps[index] === 'error'"
      >
        <summary class="cursor-pointer text-xs">
          <span class="ml-1 inline-flex flex-wrap items-center gap-2">
            <strong>挑战 {{ index + 1 }}</strong>
            <Badge
              :variant="
                run.steps[index] === 'error' ? 'destructive' : 'secondary'
              "
              class="font-normal"
            >
              {{ labels[run.steps[index]!] }}
            </Badge>
            <span class="tnum text-muted-foreground">
              期望 {{ challenge.expected_count }} 个数字 · 已识别
              {{ run.parsedCounts[index] }}
            </span>
          </span>
        </summary>
        <p class="mt-3 text-[13px] leading-relaxed text-muted-foreground">
          {{ challenge.prompt }}
        </p>
        <div class="mt-3 flex flex-col gap-2">
          <span class="text-xs font-medium">模型回答</span>
          <pre
            v-if="run.outputs[index]"
            class="max-h-56 overflow-auto whitespace-pre-wrap break-words rounded-sm border bg-muted/40 p-3 font-mono text-xs leading-relaxed"
            >{{ run.outputs[index] }}</pre>
          <p v-else class="text-xs text-muted-foreground">
            {{
              run.steps[index] === "working"
                ? "请求中，等待模型返回完整回答…"
                : "暂无回答"
            }}
          </p>
          <p
            v-if="run.stepErrors[index]"
            class="break-words text-xs text-destructive"
          >
            {{ run.stepErrors[index] }}
          </p>
        </div>
      </details>
    </section>
  </div>
</template>
