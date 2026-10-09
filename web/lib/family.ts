// 模型家族的展示配色（点 + 文字 + 底色，深浅色自适应）
export interface FamilyTone {
  dot: string
  chip: string
}

const FAMILY_TONES: Record<string, FamilyTone> = {
  gpt: {
    dot: 'bg-emerald-500',
    chip: 'border-emerald-500/30 bg-emerald-500/10 text-emerald-700 dark:text-emerald-400',
  },
  claude: {
    dot: 'bg-orange-500',
    chip: 'border-orange-500/30 bg-orange-500/10 text-orange-700 dark:text-orange-400',
  },
}

const FALLBACK_TONE: FamilyTone = {
  dot: 'bg-zinc-400',
  chip: 'border-zinc-500/30 bg-zinc-500/10 text-zinc-600 dark:text-zinc-400',
}

export function familyTone(family: string): FamilyTone {
  return FAMILY_TONES[family] ?? FALLBACK_TONE
}
