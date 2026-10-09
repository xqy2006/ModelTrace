<script setup lang="ts">
import { Toaster } from '@/components/ui/sonner'
import '@fontsource-variable/inter'
import '~/assets/css/tailwind.css'
</script>

<template>
  <!-- 灰底页面：Tab 在最顶部（框架之外），下方才是白色大框架 -->
  <div class="min-h-dvh bg-background p-5 lg:h-dvh lg:overflow-hidden">
    <div class="mx-auto flex h-full min-h-0 w-full max-w-[1600px] flex-col">
      <slot />
      <Toaster position="top-right" rich-colors close-button />
    </div>
  </div>
</template>
