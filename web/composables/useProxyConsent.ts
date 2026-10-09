// 授权只绑定当前代理地址；构建切换代理后必须重新征得同意。
const STORAGE_KEY = "modeltrace.proxy-consent.v1";

export function useProxyConsent(proxyBaseURL: string) {
  const approvedURL = useState<string>("modeltrace:proxy-consent", () => "");
  const allowed = computed(
    () => !!proxyBaseURL && approvedURL.value === proxyBaseURL,
  );

  onMounted(() => {
    try {
      approvedURL.value = localStorage.getItem(STORAGE_KEY) || "";
    } catch {
      // 存储不可用时仅在当前页面会话内记住授权。
    }
  });

  function allow() {
    approvedURL.value = proxyBaseURL;
    try {
      localStorage.setItem(STORAGE_KEY, proxyBaseURL);
    } catch {
      // 不持久化密钥、模型 ID 或挑战文本。
    }
  }

  function revoke() {
    approvedURL.value = "";
    try {
      localStorage.removeItem(STORAGE_KEY);
    } catch {
      // 当前会话仍立即撤销授权。
    }
  }

  return { allowed, allow, revoke };
}
