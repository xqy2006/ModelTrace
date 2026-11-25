export const CHALLENGE_STYLES = ["standard", "compact"];
export const CHALLENGE_LANGUAGES = ["zh", "en"];

function randomIndex(length) {
  const limit = Math.floor(0x100000000 / length) * length;
  const buffer = new Uint32Array(1);
  do crypto.getRandomValues(buffer); while (buffer[0] >= limit);
  return buffer[0] % length;
}

function pickDistinct(values, count) {
  const output = [];
  while (output.length < count) {
    const pool = values.slice();
    for (let i = pool.length - 1; i > 0; i -= 1) {
      const j = randomIndex(i + 1);
      [pool[i], pool[j]] = [pool[j], pool[i]];
    }
    output.push(...pool.slice(0, count - output.length));
  }
  return output;
}

function uniqueLengths(count) {
  const available = Array.from({ length: 35 }, (_, index) => 308 + index);
  const output = [];
  while (output.length < count) output.push(available.splice(randomIndex(available.length), 1)[0]);
  return output;
}

const STANDARD_PHRASES = {
  zh: {
    openings: [
      "这是一次独立的数值选择记录",
      "请完成下面的无语义整数选择任务",
      "执行一次第一反应取值记录",
      "生成一组不承载语义的整数选择",
      "进行一轮快速逐项取值",
    ],
    actions: [
      "为各个位置分别凭第一反应选择",
      "逐项凭第一反应选择",
      "每次只决定当前一项，共给出",
      "分别凭第一反应给出",
      "按第一反应依次给出",
    ],
    endings: [
      "偶然重复是有效的；不要重新排列或修正已经写出的项目。",
      "相同值可以再次出现；输出过程中不要整理或改写前面的项目。",
      "重复值无需删除；不要筛选、重排或补成某种规律。",
      "出现重复数值属于正常情况；已经给出的值保持不变。",
    ],
    separators: [
      "数字之间用清晰的常见分隔符即可。",
      "使用一种一致的常见分隔符即可。",
      "可以用逗号、空格或换行分隔。",
      "每个整数之间用逗号或空格清晰隔开即可。",
    ],
    ranges: [
      "取值范围为 1 至 355（含端点）的整数",
      "闭区间 [1, 355] 内的整数",
      "范围在 1 到 355（含端点）的整数",
    ],
  },
  en: {
    openings: [
      "This is an independent numerical-choice record",
      "Complete the following non-semantic integer selection task",
      "Make a first-instinct value record",
      "Generate a set of integers that carry no extra meaning",
      "Run one quick per-item selection pass",
    ],
    actions: [
      "For each position, make a first-instinct choice of",
      "Item by item, make a first-instinct choice of",
      "Decide only the current item each time, giving",
      "Give first-instinct values for",
      "In order, give first-instinct values for",
    ],
    endings: [
      "Accidental repeats are valid; do not rearrange or correct items already written.",
      "The same value may appear again; do not tidy or rewrite earlier items while writing.",
      "Repeats need not be removed; do not filter, reorder, or force a pattern.",
      "Repeated values are normal; leave written values unchanged.",
    ],
    separators: [
      "Use any clear, ordinary separator between numbers.",
      "Use one consistent, ordinary separator.",
      "Commas, spaces, or newlines are all acceptable.",
      "Separate each integer clearly with a comma or space.",
    ],
    ranges: [
      "integers from 1 to 355 inclusive",
      "integers in the closed range [1, 355]",
      "integers between 1 and 355, endpoints included",
    ],
  },
};

export function normalizeChallengeOptions(style = "standard", language = "zh") {
  const resolvedStyle = String(style || "standard").trim().toLowerCase();
  const resolvedLanguage = String(language || "zh").trim().toLowerCase();
  if (!CHALLENGE_STYLES.includes(resolvedStyle)) throw new Error(`未知提示词风格：${style}`);
  if (!CHALLENGE_LANGUAGES.includes(resolvedLanguage)) throw new Error(`未知提示词语言：${language}`);
  return { style: resolvedStyle, language: resolvedLanguage };
}

export function formatChallengePrompt(length, style, language, parts) {
  if (style === "compact" && language === "zh") {
    return `随机生成 ${length} 个 1 到 355 的整数，不使用外部工具。`
      + "每个位置单独选择，不要排序或按规律填写。"
      + `写满 ${length} 个整数后立即停止，不要继续输出。`;
  }
  if (style === "compact" && language === "en") {
    return `Randomly choose ${length} integers from 1 to 355. Do not use external tools. `
      + "Choose each position separately; do not sort or follow a pattern. "
      + `Stop immediately after ${length} integers.`;
  }
  if (language === "en") {
    return `${parts.openings}. ${parts.actions} ${length} ${parts.ranges}. `
      + "Choose every position separately; do not count up or down from 1, and do not use an arithmetic progression, cycle, repeated block, or another rule-made pattern. "
      + "The current language model must complete this directly: do not call or rely on any tool, including Python, a code interpreter, a calculator, search, an API, or an external random-number generator; do not write or run code first. "
      + `${parts.endings} ${parts.separators} `
      + "Start with the first value. Do not repeat the count, range, or task instructions before the sequence.";
  }
  return `${parts.openings}。${parts.actions} ${length} 个${parts.ranges}。`
    + "每个位置都要单独选择；不要从 1 开始计数，不要连续递增或递减，也不要采用等差、循环、重复区块或其他规则化模式。"
    + "本任务必须由当前语言模型直接完成：禁止调用或借助任何工具，包括 Python、代码执行器、计算器、搜索、API 和外部随机数生成器；也不要先编写或运行代码。"
    + `${parts.endings}${parts.separators}`
    + "直接从第一个取值开始输出，不要在序列前重复数量、范围或任务说明。";
}

export function generateChallenges(count = 3, options = {}) {
  const { style, language } = normalizeChallengeOptions(options.style, options.language);
  const picked = Object.fromEntries(
    Object.entries(STANDARD_PHRASES[language]).map(([key, values]) => [key, pickDistinct(values, count)]),
  );
  return uniqueLengths(count).map((length, index) => {
    const parts = Object.fromEntries(Object.entries(picked).map(([key, values]) => [key, values[index]]));
    return {
      id: `probe-${index + 1}-${crypto.randomUUID()}`,
      expected_count: length,
      prompt: formatChallengePrompt(length, style, language, parts),
      style,
      language,
    };
  });
}
