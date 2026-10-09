// 「开始配置」引导弹窗的全局开关：手动检测区复制提示词后可直接唤起服务商配置
export function useProviderOnboarding() {
  const configOpen = useState<boolean>('modeltrace:provider-config:open', () => false)
  const autoStart = useState<boolean>('modeltrace:provider-config:auto-start', () => false)

  function openConfig(startImmediately: boolean) {
    autoStart.value = startImmediately
    configOpen.value = true
  }

  return { configOpen, autoStart, openConfig }
}
