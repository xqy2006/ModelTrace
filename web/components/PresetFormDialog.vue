<script setup lang="ts">
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import type { ApiType, EndpointPreset } from "@/composables/usePresets";

const props = defineProps<{ open: boolean; preset: EndpointPreset | null }>();
const emit = defineEmits<{
  "update:open": [value: boolean];
  save: [input: Omit<EndpointPreset, "id">];
}>();
const form = reactive({
  name: "",
  baseUrl: "",
  apiKey: "",
  model: "",
  apiType: "chat" as ApiType,
});
const errors = reactive<Partial<Record<keyof typeof form, string>>>({});

watch(
  () => props.open,
  (open) => {
    if (!open) return;
    const source = props.preset;
    Object.assign(form, {
      name: source?.name ?? "",
      baseUrl: source?.baseUrl ?? "",
      apiKey: source?.apiKey ?? "",
      model: source?.model ?? "",
      apiType: source?.apiType ?? "chat",
    });
    Object.keys(errors).forEach(
      (key) => delete errors[key as keyof typeof form],
    );
  },
);

function submit() {
  Object.keys(errors).forEach((key) => delete errors[key as keyof typeof form]);
  if (!form.name.trim()) errors.name = "请填写服务商名称";
  if (!form.baseUrl.trim()) errors.baseUrl = "请填写 Endpoint";
  else {
    try {
      const url = new URL(form.baseUrl.trim());
      if (
        !["http:", "https:"].includes(url.protocol) ||
        !url.hostname ||
        url.username ||
        url.password ||
        url.search ||
        url.hash
      )
        throw new Error();
      if (/\/(chat\/completions|responses)\/?$/.test(url.pathname)) {
        errors.baseUrl =
          "请填写 API 根地址（例如 /v1），不要包含 /chat/completions 或 /responses";
      }
    } catch {
      errors.baseUrl =
        "请填写有效的 HTTP(S) API 根地址，不含凭据、查询参数或锚点";
    }
  }
  if (!form.apiKey.trim()) errors.apiKey = "请填写 API Key";
  if (!form.model.trim()) errors.model = "请填写模型 ID";
  if (Object.keys(errors).length) return;
  emit("save", {
    name: form.name.trim(),
    baseUrl: form.baseUrl.trim().replace(/\/+$/, ""),
    apiKey: form.apiKey.trim(),
    model: form.model.trim(),
    apiType: form.apiType,
    temperature: props.preset?.temperature ?? null,
  });
  emit("update:open", false);
}
</script>

<template>
  <Dialog :open="open" @update:open="emit('update:open', $event)">
    <DialogContent class="max-h-[90dvh] overflow-y-auto sm:max-w-lg">
      <DialogHeader>
        <DialogTitle>{{ preset ? "编辑服务商" : "添加服务商" }}</DialogTitle>
        <DialogDescription>
          四项均为必填，配置仅保存在当前浏览器。直连只发送至服务商；如果你选择代理测试，密钥和挑战会经过代理服务器。
        </DialogDescription>
      </DialogHeader>
      <form class="flex flex-col gap-4" novalidate @submit.prevent="submit">
        <div class="flex flex-col gap-2" :data-invalid="!!errors.name">
          <Label for="preset-name">服务商名称 *</Label>
          <Input
            id="preset-name"
            v-model="form.name"
            required
            placeholder="例如：OpenAI 官方"
            :aria-invalid="!!errors.name"
            aria-describedby="name-error"
          />
          <p
            v-if="errors.name"
            id="name-error"
            class="text-xs text-destructive"
          >
            {{ errors.name }}
          </p>
        </div>
        <div class="flex flex-col gap-2" :data-invalid="!!errors.baseUrl">
          <Label for="preset-base-url">Endpoint *</Label>
          <Input
            id="preset-base-url"
            v-model="form.baseUrl"
            required
            type="url"
            placeholder="https://api.openai.com/v1"
            :aria-invalid="!!errors.baseUrl"
            aria-describedby="endpoint-hint endpoint-error"
          />
          <p id="endpoint-hint" class="text-xs text-muted-foreground">
            API 根地址；SDK 会根据协议自动追加请求路径。
          </p>
          <p
            v-if="errors.baseUrl"
            id="endpoint-error"
            class="text-xs text-destructive"
          >
            {{ errors.baseUrl }}
          </p>
        </div>
        <div class="flex flex-col gap-2" :data-invalid="!!errors.apiKey">
          <Label for="preset-api-key">API Key *</Label>
          <Input
            id="preset-api-key"
            v-model="form.apiKey"
            required
            type="password"
            autocomplete="off"
            placeholder="sk-..."
            :aria-invalid="!!errors.apiKey"
            aria-describedby="key-error"
          />
          <p
            v-if="errors.apiKey"
            id="key-error"
            class="text-xs text-destructive"
          >
            {{ errors.apiKey }}
          </p>
        </div>
        <div class="flex flex-col gap-2" :data-invalid="!!errors.model">
          <Label for="preset-model">模型 ID *</Label>
          <Input
            id="preset-model"
            v-model="form.model"
            required
            placeholder="例如：gpt-4o"
            :aria-invalid="!!errors.model"
            aria-describedby="model-error"
          />
          <p
            v-if="errors.model"
            id="model-error"
            class="text-xs text-destructive"
          >
            {{ errors.model }}
          </p>
        </div>
        <details
          class="rounded-md border p-3"
          :open="form.apiType === 'responses'"
        >
          <summary class="cursor-pointer text-xs text-muted-foreground">
            请求协议（默认 Chat Completions）
          </summary>
          <div class="mt-3 flex flex-col gap-2">
            <Label for="preset-protocol">API 协议</Label>
            <Select v-model="form.apiType">
              <SelectTrigger id="preset-protocol"
                ><SelectValue
              /></SelectTrigger>
              <SelectContent
                ><SelectGroup>
                  <SelectItem value="chat">Chat Completions</SelectItem>
                  <SelectItem value="responses">Responses API</SelectItem>
                </SelectGroup></SelectContent
              >
            </Select>
          </div>
        </details>
        <DialogFooter>
          <Button
            type="button"
            variant="outline"
            @click="emit('update:open', false)"
            >取消</Button
          >
          <Button type="submit">保存</Button>
        </DialogFooter>
      </form>
    </DialogContent>
  </Dialog>
</template>
