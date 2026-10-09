<script setup lang="ts">
import { ChevronDown, Loader2, Pencil, Play, RotateCcw, Trash2 } from 'lucide-vue-next'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Progress } from '@/components/ui/progress'
import { presetLabel, type EndpointPreset } from '@/composables/usePresets'
import type { RunStatus, StepState } from '@/composables/useApiTest'
import { percent } from '@/lib/format'
import { cn } from '@/lib/utils'

const props = defineProps<{ preset: EndpointPreset }>()
const emit = defineEmits<{ edit: []; remove: [] }>()

const { runStates, batchRunning, isRunning, runPreset } = useApiTest()
const { bank } = useBank()

const expanded = ref(false)

const run = computed(() => runStates.value[props.preset.id])
const running = computed(() => isRunning(props.preset.id))

const STEP_LABELS: Record<StepState, string> = {
  pending: '等待',
  working: '请求中',
  done: '有效',
  invalid: '数字不足',
  error: '失败',
  skipped: '未调用',
}

const stepClass = (step: StepState) =>
  ({
    pending: 'border-border text-muted-foreground',
    working: 'border-sky-500/30 bg-sky-500/10 text-sky-700 dark:text-sky-400',
    done: 'border-emerald-500/30 bg-emerald-500/10 text-emerald-700 dark:text-emerald-400',
    invalid: 'border-amber-500/30 bg-amber-500/10 text-amber-700 dark:text-amber-400',
    error: 'border-red-500/30 bg-red-500/10 text-red-700 dark:text-red-400',
    skipped: 'border-border text-muted-foreground line-through',
  })[step]

const STATUS_DOT: Record<RunStatus, string> = {
  running: 'bg-sky-500',
  success: 'bg-emerald-500',
  failed: 'bg-red-500',
}

async function handleTest() {
  if (!bank.value) return
  expanded.value = true
  await runPreset(props.preset)
}

function maskApiKey(key: string): string {
  if (key.length <= 8) return '••••••••'
  return `${key.slice(0, 4)}…${key.slice(-4)}`
}
</script>

<template>
  <div class="rounded-2xl border bg-card shadow-sm transition-colors hover:border-foreground/20">
    <!-- 第一行：服务商信息 -->
    <div class="flex flex-wrap items-center gap-x-4 gap-y-3 p-5">
      <span class="relative flex size-2.5 shrink-0">
        <span
          v-if="running"
          class="absolute inline-flex size-full animate-ping rounded-full bg-sky-400 opacity-60"
        />
        <span
          :class="cn(
            'relative inline-flex size-2.5 rounded-full',
            run ? STATUS_DOT[run.status] : 'bg-muted-foreground/30',
          )"
        />
      </span>

      <div class="min-w-0">
        <div class="flex flex-wrap items-center gap-2">
          <strong class="text-sm font-semibold">{{ presetLabel(preset) }}</strong>
          <Badge variant="outline" class="font-normal">
            {{ preset.apiType === 'responses' ? 'Responses' : 'Chat' }}
          </Badge>
          <Badge variant="secondary" class="font-normal">
            <code>{{ preset.model }}</code>
          </Badge>
        </div>
        <span class="mt-0.5 block truncate text-xs text-muted-foreground" :title="preset.baseUrl">
          {{ preset.baseUrl }} · 密钥 {{ maskApiKey(preset.apiKey)
          }}{{ preset.temperature !== null ? ` · 温度 ${preset.temperature}` : '' }}
        </span>
      </div>

      <!-- 右侧：状态 / 结果 -->
      <div class="ml-auto flex items-center gap-2">
        <div v-if="run?.status === 'success' && run.result" class="mr-1 text-right">
          <div class="text-sm font-extrabold uppercase leading-tight tracking-tight">
            {{ run.result.prediction_name }}
          </div>
          <div class="tnum text-xs text-muted-foreground">
            {{ percent(run.result.probability) }}
          </div>
        </div>
        <div v-else-if="running" class="mr-1 flex items-center gap-2 text-xs text-muted-foreground">
          <Loader2 class="size-3.5 animate-spin" />
          <span class="tnum">有效 {{ run?.validCount ?? 0 }}/3</span>
        </div>

        <Button
          v-if="!run || run.status === 'failed'"
          size="sm"
          :disabled="running || batchRunning"
          @click="handleTest"
        >
          <Play data-icon="inline-start" />
          {{ run?.status === 'failed' ? '重试' : '开始测试' }}
        </Button>
        <Button
          v-else-if="!running"
          size="sm"
          variant="outline"
          :disabled="batchRunning"
          @click="handleTest"
        >
          <RotateCcw data-icon="inline-start" />
          重新测试
        </Button>

        <Button variant="ghost" size="icon" :disabled="running" aria-label="编辑" @click="emit('edit')">
          <Pencil class="size-4" />
        </Button>
        <Button variant="ghost" size="icon" :disabled="running" aria-label="删除" @click="emit('remove')">
          <Trash2 class="size-4" />
        </Button>
        <Button
          v-if="run"
          variant="ghost"
          size="icon"
          :aria-label="expanded ? '收起' : '展开'"
          @click="expanded = !expanded"
        >
          <ChevronDown :class="cn('size-4 transition-transform', expanded && 'rotate-180')" />
        </Button>
      </div>
    </div>

    <!-- 展开区：挑战明细 + 进度 + 分布图 -->
    <div v-if="run && expanded" class="flex flex-col gap-4 border-t p-5">
      <div class="flex flex-col gap-2">
        <div
          v-for="(challenge, index) in run.challenges"
          :key="challenge.id"
          class="flex items-start gap-3 rounded-xl bg-muted/50 p-3"
        >
          <Badge
            variant="outline"
            :class="cn('mt-0.5 shrink-0 font-normal', stepClass(run.steps[index]))"
          >
            {{ STEP_LABELS[run.steps[index]] }}
          </Badge>
          <pre
            class="line-clamp-2 min-w-0 flex-1 whitespace-pre-wrap font-mono text-[11px] leading-relaxed text-muted-foreground"
          >{{ challenge.prompt }}</pre>
        </div>
      </div>

      <div class="flex items-center gap-3">
        <Progress :model-value="(run.validCount / 3) * 100" class="h-1.5 flex-1" />
        <span class="tnum text-xs text-muted-foreground">有效 {{ run.validCount }}/3</span>
      </div>
      <p
        class="text-xs"
        :class="run.status === 'failed' ? 'text-destructive' : 'text-muted-foreground'"
      >
        {{ run.message }}
      </p>

      <template v-if="run.result">
        <div class="flex flex-col gap-2 rounded-xl border p-4">
          <div class="flex items-baseline justify-between gap-2">
            <span class="text-xs font-medium uppercase tracking-widest text-muted-foreground">
              候选分布
            </span>
            <span class="text-sm">
              最可能
              <strong class="font-extrabold uppercase">{{ run.result.prediction_name }}</strong>
              <span class="tnum text-muted-foreground"> {{ percent(run.result.probability) }}</span>
            </span>
          </div>
          <ProbBarChart :results="run.result.results" />
        </div>
      </template>
    </div>
  </div>
</template>
