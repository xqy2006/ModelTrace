<script setup lang="ts">
import { Github, Moon, PenLine, Sun, Zap } from "lucide-vue-next";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

const mode = ref("manual");
const { isDark, toggle } = useTheme();
</script>

<template>
  <main class="flex min-h-0 flex-1 flex-col">
    <Tabs v-model="mode" class="flex min-h-0 flex-1 flex-col">
      <!-- 模式切换仍位于大框架外，不引入导航栏或 Logo。 -->
      <div class="flex shrink-0 items-center justify-between gap-3">
        <TabsList
          class="h-auto justify-start gap-3 bg-transparent p-0"
          aria-label="测试模式"
        >
          <TabsTrigger
            value="manual"
            class="border bg-secondary px-3 py-2 data-[state=active]:bg-primary data-[state=active]:text-primary-foreground"
          >
            <span class="flex items-center gap-1.5"
              ><PenLine class="size-4 stroke-[1.5px]" />手动测试</span
            >
          </TabsTrigger>
          <TabsTrigger
            value="auto"
            class="border bg-secondary px-3 py-2 data-[state=active]:bg-primary data-[state=active]:text-primary-foreground"
          >
            <span class="flex items-center gap-1.5"
              ><Zap class="size-4 stroke-[1.5px]" />自动测试</span
            >
          </TabsTrigger>
        </TabsList>
        <div class="flex items-center gap-2">
          <Button
            variant="outline"
            size="icon"
            aria-label="切换主题"
            @click="toggle"
          >
            <Sun v-if="isDark" /><Moon v-else />
          </Button>
          <Button variant="outline" size="icon" as-child>
            <a
              href="https://github.com/Hanmo123/ModelTrace"
              target="_blank"
              rel="noopener noreferrer"
              aria-label="GitHub 项目仓库"
              ><Github
            /></a>
          </Button>
        </div>
      </div>
      <!-- 保留面板实例，切换模式不清空输入，也不中断自动测试。 -->
      <TabsContent
        v-show="mode === 'manual'"
        value="manual"
        force-mount
        class="mt-0 flex min-h-0 flex-1 flex-col"
      >
        <ManualSection />
      </TabsContent>
      <TabsContent
        v-show="mode === 'auto'"
        value="auto"
        force-mount
        class="mt-0 flex min-h-0 flex-1 flex-col"
      >
        <AutoSection />
      </TabsContent>
    </Tabs>
  </main>
</template>
