<script setup lang="ts">
import { ClipboardCopy, RefreshCw, Terminal } from "lucide-vue-next";
import { toast } from "vue-sonner";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import type { EndpointPreset } from "@/composables/usePresets";
import {
  terminalCommand,
  terminalKeySetup,
  type OutputFormat,
  type TerminalShell,
} from "@/lib/terminal-commands";
import { parseNumbers } from "@/lib/fingerprint";

const props = defineProps<{ open: boolean; preset: EndpointPreset | null }>();
const emit = defineEmits<{ "update:open": [value: boolean] }>();
const { sessions, start, parse } = useTerminalTest();
const shell = ref<"posix" | "powershell" | "cmd">("posix");
const format = ref<OutputFormat>("text");
const session = computed(() =>
  props.preset ? sessions.value[props.preset.id] : null,
);
const commandShell = computed<TerminalShell>(() =>
  shell.value === "posix" ? "posix" : "powershell",
);
const setup = computed(() => terminalKeySetup(commandShell.value));

watch(
  () => props.open,
  (open) => {
    if (open && props.preset && !sessions.value[props.preset.id])
      start(props.preset.id);
  },
);

let timer: ReturnType<typeof setTimeout> | undefined;
function onAnswerInput() {
  if (timer) clearTimeout(timer);
  timer = setTimeout(() => {
    if (props.preset) parse(props.preset.id, props.preset.apiKey);
  }, 500);
}
onUnmounted(() => {
  if (timer) clearTimeout(timer);
});

async function copy(value: string, isKey = false) {
  try {
    await navigator.clipboard.writeText(value);
    toast.success(
      isKey ? "密钥已复制：粘贴到终端后请清除剪贴板" : "已复制到剪贴板",
    );
  } catch {
    toast.error("剪贴板不可用，请手动复制");
  }
}

function restart() {
  if (!props.preset) return;
  start(props.preset.id);
  toast.info("已生成新挑战，之前粘贴的回答已清空");
}
</script>

<template>
  <Dialog :open="open" @update:open="emit('update:open', $event)">
    <DialogContent class="max-h-[90dvh] overflow-y-auto sm:max-w-4xl">
      <DialogHeader>
        <DialogTitle>终端 curl 测试 · {{ preset?.name }}</DialogTitle>
        <DialogDescription>
          浏览器无法跨域时，在自己的终端直连服务商。仅复制命令不会发起请求；密钥不写入命令或仓库。
          命令将提取回答并尝试复制到本机剪贴板，把输出粘贴到对应题目即可本地计算。执行时密钥可能短暂出现在 curl 进程参数中。
        </DialogDescription>
      </DialogHeader>
      <div v-if="preset && session" class="flex flex-col gap-4">
        <div class="flex flex-wrap items-end gap-3">
          <div class="flex min-w-40 flex-col gap-1.5">
            <Label for="terminal-shell">终端</Label>
            <Select v-model="shell"
              ><SelectTrigger id="terminal-shell"
                ><SelectValue
              /></SelectTrigger>
              <SelectContent
                ><SelectGroup>
                  <SelectItem value="posix">Linux / macOS</SelectItem>
                  <SelectItem value="powershell">Windows PowerShell</SelectItem>
                  <SelectItem value="cmd">Windows CMD</SelectItem>
                </SelectGroup></SelectContent
              >
            </Select>
          </div>
          <div class="flex min-w-48 flex-col gap-1.5">
            <Label for="terminal-format">命令输出</Label>
            <Select v-model="format"
              ><SelectTrigger id="terminal-format"
                ><SelectValue
              /></SelectTrigger>
              <SelectContent
                ><SelectGroup>
                  <SelectItem value="text"
                    >提取回答文本{{
                      shell === "posix" ? "（需 Python 3）" : ""
                    }}</SelectItem
                  >
                  <SelectItem value="json">原始 JSON（页面会解析）</SelectItem>
                </SelectGroup></SelectContent
              ></Select
            >
          </div>
          <Button variant="outline" size="sm" @click="restart"
            ><RefreshCw data-icon="inline-start" />新一轮挑战</Button
          >
        </div>
        <p v-if="shell === 'cmd'" class="text-xs text-muted-foreground">
          请先在 CMD 输入 <code>powershell</code> 并回车，再复制下方 PowerShell
          命令。CMD 无需另装程序。
        </p>
        <div class="flex flex-col gap-2 rounded-md border p-3">
          <div class="flex flex-wrap items-center justify-between gap-2">
            <strong class="text-sm"
              >① 当前终端会话输入 API Key（只需一次）</strong
            >
            <div class="flex gap-2">
              <Button size="sm" variant="outline" @click="copy(setup)"
                ><ClipboardCopy data-icon="inline-start" />复制设置命令</Button
              >
              <Button
                size="sm"
                variant="ghost"
                @click="copy(preset.apiKey, true)"
                >复制已保存密钥</Button
              >
            </div>
          </div>
          <pre
            class="overflow-x-auto rounded-sm bg-muted/60 p-2 font-mono text-xs"
            >{{ setup }}</pre>
          <p class="text-xs text-muted-foreground">
            先执行设置命令，按提示粘贴密钥并回车；输入不显示、不会进入命令历史。请在<strong>同一个终端窗口</strong>执行下面的命令。
          </p>
        </div>
        <p class="text-xs text-muted-foreground">
          ②
          每道题复制一条命令。命令包含独立提示词，输出会自动尝试写入本机剪贴板；手动粘贴回对应输入框。任意一份有效回答即可出结果。
        </p>
        <div
          v-for="(challenge, index) in session.challenges"
          :key="challenge.id"
          class="flex flex-col gap-2 rounded-md border p-3"
        >
          <div class="flex flex-wrap items-center justify-between gap-2">
            <strong class="text-sm"
              >挑战 {{ index + 1 }} ·
              {{ challenge.expected_count }} 个数字</strong
            >
            <Button
              variant="outline"
              size="sm"
              @click="
                copy(
                  terminalCommand(
                    preset,
                    challenge.prompt,
                    commandShell,
                    format,
                  ),
                )
              "
            >
              <Terminal data-icon="inline-start" />复制 curl 命令
            </Button>
          </div>
          <pre
            class="max-h-24 overflow-auto whitespace-pre-wrap break-all rounded-sm bg-muted/60 p-2 font-mono text-[11px]"
            >{{
              terminalCommand(preset, challenge.prompt, commandShell, format)
            }}</pre>
          <Textarea
            v-model="session.raw[index]"
            :aria-label="`挑战 ${index + 1} 终端输出`"
            class="min-h-20 font-mono text-xs"
            placeholder="把终端提取的回答文本或原始 JSON 粘贴到这里"
            @input="onAnswerInput"
          />
          <p v-if="session.errors[index]" class="text-xs text-destructive">
            {{ session.errors[index] }}
          </p>
          <p
            v-else-if="session.raw[index]"
            class="text-xs text-muted-foreground"
          >
            已识别
            {{ parseNumbers(session.outputs[index] || "").length }} 个数字（最低
            {{ Math.max(80, Math.ceil(challenge.expected_count * 0.55)) }}）
          </p>
        </div>
        <p v-if="session.result" class="text-sm font-semibold uppercase">
          当前结果：{{ session.result.prediction_name }} ·
          {{ (session.result.probability * 100).toFixed(1) }}%（有效回答
          {{ session.result.used_outputs }}/3）
        </p>
        <p class="text-xs text-muted-foreground">
          如果 Python 3 不可用，请切换「原始
          JSON」命令。终端执行会消耗服务商额度；不要将包含密钥的终端记录或剪贴板内容分享给他人。
        </p>
      </div>
    </DialogContent>
  </Dialog>
</template>
