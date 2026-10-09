<script setup lang="ts">
import type { ModelResult } from '@/lib/fingerprint'
import { percent } from '@/lib/format'
import { cn } from '@/lib/utils'

const props = withDefaults(defineProps<{ results: ModelResult[]; limit?: number }>(), { limit: 6 })

const top = computed(() => props.results.slice(0, props.limit))
const gridLines = [100, 75, 50, 25, 0]

/** 缩短模型名用于横轴标签 */
const shortName = (name: string) => name.replace(/^(gpt|claude)-/i, '').replace(/-\d{8}$/, '')
/** 仅给头名与概率 >= 5% 的柱子标注数值，避免拥挤截断 */
const labeled = (item: ModelResult, index: number) => index === 0 || item.probability >= 0.05
</script>

<template>
  <div class="flex flex-col gap-1">
    <div class="flex h-44 gap-2">
      <!-- 纵轴刻度 -->
      <div class="relative w-8 shrink-0" aria-hidden="true">
        <span
          v-for="line in gridLines"
          :key="line"
          class="tnum absolute right-0 translate-y-1/2 text-[10px] text-muted-foreground"
          :style="{ bottom: `${line}%` }"
        >{{ line }}%</span>
      </div>

      <!-- 绘图区 -->
      <div class="relative min-w-0 flex-1">
        <div
          v-for="line in gridLines"
          :key="line"
          class="absolute inset-x-0 border-t border-dashed border-border"
          :style="{ bottom: `${line}%` }"
          aria-hidden="true"
        />
        <div class="flex h-full items-end gap-2">
          <div
            v-for="(item, index) in top"
            :key="item.model"
            class="group relative flex h-full min-w-0 flex-1 flex-col justify-end"
            :title="`${item.display_name} · ${percent(item.probability)}`"
          >
            <span
              v-if="labeled(item, index)"
              class="tnum mb-1 whitespace-nowrap text-center text-[10px] font-medium"
              :class="index === 0 ? 'text-foreground' : 'text-muted-foreground'"
            >
              {{ percent(item.probability) }}
            </span>
            <div
              :class="cn(
                'w-full rounded-t-md transition-all duration-500',
                index === 0 ? 'bg-[hsl(var(--chart-1))]' : 'bg-[hsl(var(--chart-2))] group-hover:opacity-80',
              )"
              :style="{ height: `${Math.max(item.probability * 100, 1.2)}%` }"
            />
          </div>
        </div>
      </div>
    </div>

    <!-- 横轴标签 -->
    <div class="flex gap-2">
      <div class="w-8 shrink-0" aria-hidden="true" />
      <div class="flex min-w-0 flex-1 gap-2">
        <span
          v-for="(item, index) in top"
          :key="item.model"
          :class="
            cn(
              'min-w-0 flex-1 truncate text-center text-[10px]',
              index === 0 ? 'font-semibold text-foreground' : 'text-muted-foreground',
            )
          "
          >{{ shortName(item.display_name) }}</span
        >
      </div>
    </div>
    <p v-if="results.length > limit" class="text-right text-[10px] text-muted-foreground">
      仅展示前 {{ limit }} 个候选，共 {{ results.length }} 个
    </p>
  </div>
</template>
