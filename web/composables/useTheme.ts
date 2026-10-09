import { useDark, useToggle } from '@vueuse/core'

// 主题切换：class 策略写入 <html>，偏好由 vueuse 持久化到 localStorage
export function useTheme() {
  const isDark = useDark({
    selector: 'html',
    attribute: 'class',
    valueDark: 'dark',
    valueLight: '',
    storageKey: 'modeltrace.theme',
  })
  const toggle = useToggle(isDark)
  return { isDark, toggle }
}
