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

export function generateChallenges(count = 3) {
  const openings = [
    "这是一次独立的数值选择记录",
    "请完成下面的无语义整数选择任务",
    "执行一次第一反应取值记录",
    "生成一组不承载语义的整数选择",
    "进行一轮快速逐项取值",
  ];
  const actions = [
    "为各个位置分别凭第一反应选择",
    "逐项凭第一反应选择",
    "每次只决定当前一项，共给出",
    "分别凭第一反应给出",
    "按第一反应依次给出",
  ];
  const endings = [
    "偶然重复是有效的；不要重新排列或修正已经写出的项目。",
    "相同值可以再次出现；输出过程中不要整理或改写前面的项目。",
    "重复值无需删除；不要筛选、重排或补成某种规律。",
    "出现重复数值属于正常情况；已经给出的值保持不变。",
  ];
  const separators = [
    "数字之间用清晰的常见分隔符即可。",
    "使用一种一致的常见分隔符即可。",
    "可以用逗号、空格或换行分隔。",
    "每个整数之间用逗号或空格清晰隔开即可。",
  ];
  const rangePhrases = [
    "取值范围为 1 至 355（含端点）的整数",
    "闭区间 [1, 355] 内的整数",
    "范围在 1 到 355（含端点）的整数",
  ];
  const pickedOpenings = pickDistinct(openings, count);
  const pickedActions = pickDistinct(actions, count);
  const pickedEndings = pickDistinct(endings, count);
  const pickedSeparators = pickDistinct(separators, count);
  const pickedRanges = pickDistinct(rangePhrases, count);
  return uniqueLengths(count).map((length, index) => ({
    id: `probe-${index + 1}-${crypto.randomUUID()}`,
    expected_count: length,
    prompt: `${pickedOpenings[index]}。${pickedActions[index]} ${length} 个${pickedRanges[index]}。`
      + "每个位置都要单独选择；不要从 1 开始计数，不要连续递增或递减，也不要采用等差、循环、重复区块或其他规则化模式。"
      + "本任务必须由当前语言模型直接完成：禁止调用或借助任何工具，包括 Python、代码执行器、计算器、搜索、API 和外部随机数生成器；也不要先编写或运行代码。"
      + `${pickedEndings[index]}${pickedSeparators[index]}`
      + "直接从第一个取值开始输出，不要在序列前重复数量、范围或任务说明。",
  }));
}
