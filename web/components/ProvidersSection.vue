<script setup lang="ts">
import { Loader2, Plus, Zap } from 'lucide-vue-next'
import { toast } from 'vue-sonner'
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from '@/components/ui/alert-dialog'
import { Button } from '@/components/ui/button'
import { presetLabel, type EndpointPreset } from '@/composables/usePresets'
import { percent } from '@/lib/format'

const { presets, addPreset, updatePreset, removePreset } = usePresets()
const { runStates, batchRunning, runPreset, runBatch, clearRun } = useApiTest()
const { bank, load: loadBank } = useBank()
const { configOpen, autoStart } = useProviderOnboarding()

const dialogOpen = ref(false)
const editingPreset = ref<EndpointPreset | null>(null)
const deletingPreset = ref<EndpointPreset | null>(null)

// 复制提示词后「开始配置」唤起同一弹窗
watch(configOpen, (open) => {
  if (open) {
    editingPreset.value = null
    dialogOpen.value = true
  }
})
watch(dialogOpen, (open) => {
  if (!open) configOpen.value = false
})

function openCreate() {
  autoStart.value = false
  editingPreset.value = null
  dialogOpen.value = true
}

function openEdit(preset: EndpointPreset) {
  autoStart.value = false
  editingPreset.value = preset
  dialogOpen.value = true
}

async function handleSave(input: Omit<EndpointPreset, 'id'>) {
  if (editingPreset.value) {
    updatePreset(editingPreset.value.id, input)
    toast.success(`「${presetLabel(input)}」已更新`)
    return
  }
  const preset = addPreset(input)
  const label = presetLabel(preset)
  if (autoStart.value && bank.value) {
    toast.success(`「${label}」已保存，开始自动检测`)
    const state = await runPreset(preset)
    if (state.status === 'success' && state.result) {
      toast.success(
        `「${label}」检测完成：${state.result.prediction_name}（${percent(state.result.probability)}）`,
      )
    } else {
      toast.error(`「${label}」检测失败：${state.message}`)
    }
  } else {
    toast.success(`「${label}」已保存`)
  }
  autoStart.value = false
}

function confirmDelete() {
  if (!deletingPreset.value) return
  clearRun(deletingPreset.value.id)
  removePreset(deletingPreset.value.id)
  toast.success(`「${presetLabel(deletingPreset.value)}」已删除`)
  deletingPreset.value = null
}

async function handleBatch() {
  if (!bank.value) {
    toast.error('指纹库尚未加载完成，请稍后再试')
    return
  }
  toast.info(`开始批量检测 ${presets.value.length} 组服务商（并发 2）`)
  await runBatch(presets.value)
  const states = Object.values(runStates.value)
  const success = states.filter((state) => state.status === 'success').length
  toast.success(`批量检测完成：成功 ${success} 组，失败 ${states.length - success} 组`)
}

onMounted(loadBank)
</script>

<template>
  <section class="flex flex-col gap-4">
    <div class="flex flex-wrap items-center justify-between gap-3">
      <div class="flex flex-col gap-1">
        <h2 class="text-lg font-bold tracking-tight">API 自动检测</h2>
        <p class="text-sm text-muted-foreground">
          浏览器直连各服务商（无后端中转），支持 Chat Completions 与 Responses 两种协议；目标服务需允许跨域（CORS）。
        </p>
      </div>
      <div class="flex items-center gap-2">
        <Button variant="outline" @click="openCreate">
          <Plus data-icon="inline-start" />
          添加服务商
        </Button>
        <Button :disabled="!presets.length || batchRunning" @click="handleBatch">
          <Loader2 v-if="batchRunning" class="animate-spin" data-icon="inline-start" />
          <Zap v-else data-icon="inline-start" />
          {{ batchRunning ? '批量检测中…' : `批量测试全部${presets.length ? `（${presets.length}）` : ''}` }}
        </Button>
      </div>
    </div>

    <button
      v-if="!presets.length"
      type="button"
      class="flex flex-col items-center gap-2 rounded-2xl border border-dashed bg-card/60 p-10 text-center transition-colors hover:border-foreground/30 hover:bg-card"
      @click="openCreate"
    >
      <span class="brand-mark flex size-10 items-center justify-center rounded-xl">
        <Plus class="size-5" />
      </span>
      <strong class="text-sm">添加第一个服务商</strong>
      <span class="max-w-sm text-xs leading-relaxed text-muted-foreground">
        填写 Endpoint、API Key 与模型名即可开始自动检测；可添加多组后一键批量测试。所有信息只存在当前浏览器。
      </span>
    </button>

    <div v-else class="flex flex-col gap-4">
      <ProviderCard
        v-for="preset in presets"
        :key="preset.id"
        :preset="preset"
        @edit="openEdit(preset)"
        @remove="deletingPreset = preset"
      />
    </div>

    <PresetFormDialog v-model:open="dialogOpen" :preset="editingPreset" @save="handleSave" />

    <AlertDialog :open="!!deletingPreset" @update:open="deletingPreset = null">
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>删除服务商</AlertDialogTitle>
          <AlertDialogDescription>
            确定删除「{{ deletingPreset ? presetLabel(deletingPreset) : '' }}」吗？此操作不可撤销。
          </AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel>取消</AlertDialogCancel>
          <AlertDialogAction @click="confirmDelete">删除</AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  </section>
</template>
