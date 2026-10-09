import type { UnifiedBank } from '@/lib/fingerprint'

// 指纹库全局单例：仅在浏览器端加载一次（ssr: false）
export function useBank() {
  const bank = useState<UnifiedBank | null>('modeltrace:bank', () => null)
  const error = useState<string | null>('modeltrace:bank:error', () => null)
  const loading = useState<boolean>('modeltrace:bank:loading', () => false)

  const modelCount = computed(() => bank.value?.models.length ?? 0)
  const fingerprintCount = computed(
    () => bank.value?.models.reduce((sum, model) => sum + model.response_count, 0) ?? 0,
  )

  async function load() {
    if (bank.value || loading.value) return
    loading.value = true
    error.value = null
    try {
      const base = useRuntimeConfig().app.baseURL.replace(/\/?$/, '/')
      bank.value = await $fetch<UnifiedBank>(`${base}data/unified_bank.json`, { cache: 'no-cache' })
    } catch (err) {
      error.value = err instanceof Error ? err.message : '指纹库加载失败'
    } finally {
      loading.value = false
    }
  }

  return { bank, error, loading, modelCount, fingerprintCount, load }
}
