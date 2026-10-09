<script setup lang="ts">
import { AlertCircle, ScanSearch } from 'lucide-vue-next'
import { Badge } from '@/components/ui/badge'
import type { AnalysisResult } from '@/lib/fingerprint'
import { familyTone } from '@/lib/family'
import { percent } from '@/lib/format'
import { cn } from '@/lib/utils'

defineProps<{ result: AnalysisResult }>()

const diagnosisClass = (accepted: boolean) =>
  accepted
    ? 'border-emerald-500/30 bg-emerald-500/10 text-emerald-700 dark:text-emerald-400'
    : 'border-amber-500/30 bg-amber-500/10 text-amber-700 dark:text-amber-400'
</script>

<template>
  <div class="flex flex-col gap-3 lg:gap-4">
    <!-- 判定卡 -->
    <div class="rounded-2xl border bg-card p-5">
      <div class="flex items-center justify-between gap-2">
        <div class="flex items-center gap-2.5">
          <span
            class="flex size-9 items-center justify-center rounded-xl bg-muted text-muted-foreground"
          >
            <ScanSearch class="size-4" />
          </span>
          <span class="text-xs font-medium uppercase tracking-widest text-muted-foreground">
            最可能模型
          </span>
        </div>
        <span class="tnum text-xs text-muted-foreground">有效查询 {{ result.used_outputs }}/3</span>
      </div>

      <div class="mt-5 flex items-end justify-between gap-3">
        <h2 class="mb-0.5 break-all text-2xl font-extrabold uppercase leading-none tracking-tight">
          {{ result.prediction_name }}
        </h2>
        <strong class="tnum shrink-0 text-5xl font-extrabold leading-none tracking-tight">
          {{ percent(result.probability) }}
        </strong>
      </div>

      <div class="mt-4 flex flex-wrap gap-1.5">
        <Badge variant="outline" :class="cn('font-normal', familyTone(result.family_prediction).chip)">
          {{ result.family_prediction_name }} 家族 · {{ percent(result.family_probability) }}
        </Badge>
        <Badge
          v-for="item in result.diagnostics"
          :key="item.index"
          variant="outline"
          :class="cn('font-normal', diagnosisClass(item.accepted))"
        >
          挑战 {{ item.index + 1 }}：{{ item.parsed_numbers }} 个数字 ·
          {{ item.accepted ? '计入' : '忽略' }}
        </Badge>
      </div>
    </div>

    <!-- 分布卡 -->
    <div class="rounded-2xl border bg-card p-5">
      <div class="mb-4 flex items-center justify-between gap-2">
        <span class="text-xs font-medium uppercase tracking-widest text-muted-foreground">
          候选分布
        </span>
        <div class="flex items-center gap-3 text-[11px] text-muted-foreground">
          <span class="flex items-center gap-1.5">
            <i class="size-2.5 rounded-sm bg-[hsl(var(--chart-1))]" />最可能
          </span>
          <span class="flex items-center gap-1.5">
            <i class="size-2.5 rounded-sm bg-[hsl(var(--chart-2))]" />其他候选
          </span>
        </div>
      </div>
      <ProbBarChart :results="result.results" />
    </div>

    <p class="flex items-start gap-1.5 px-1 text-xs leading-relaxed text-muted-foreground">
      <AlertCircle class="mt-0.5 size-3.5 shrink-0" />
      仅对指纹库内的模型进行归因；待测模型不在库中时，任何结果都有可能。Claude Code
      的系统提示词会显著影响偏好，不建议在其中测试。
    </p>
  </div>
</template>
