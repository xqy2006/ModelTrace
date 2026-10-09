import { generateChallenges, type Challenge } from "@/lib/challenge";
import {
  analyzeGlobalOutputs,
  parseNumbers,
  type AnalysisResult,
} from "@/lib/fingerprint";
import { extractTerminalAnswer } from "@/lib/terminal-commands";

export interface TerminalSession {
  challenges: Challenge[];
  raw: string[];
  outputs: string[];
  errors: string[];
  result: AnalysisResult | null;
}

export function useTerminalTest() {
  const sessions = useState<Record<string, TerminalSession>>(
    "modeltrace:terminal-sessions",
    () => ({}),
  );
  const { bank } = useBank();

  function start(id: string) {
    const challenges = generateChallenges(3);
    sessions.value = {
      ...sessions.value,
      [id]: reactive<TerminalSession>({
        challenges,
        raw: ["", "", ""],
        outputs: ["", "", ""],
        errors: ["", "", ""],
        result: null,
      }),
    };
  }

  function clear(id: string) {
    const next = { ...sessions.value };
    delete next[id];
    sessions.value = next;
  }

  function parse(id: string, apiKey = "") {
    const session = sessions.value[id];
    if (!session) return;
    session.raw.forEach((raw, index) => {
      try {
        session.outputs[index] = extractTerminalAnswer(raw);
        session.errors[index] = "";
      } catch (error) {
        session.outputs[index] = "";
        const message = error instanceof Error ? error.message : String(error);
        session.errors[index] = (
          apiKey ? message.split(apiKey).join("[已隐藏密钥]") : message
        ).slice(0, 250);
      }
    });
    if (!bank.value) return;
    try {
      session.result = analyzeGlobalOutputs(
        session.challenges.map((challenge, index) => ({
          text: session.outputs[index] || "",
          expected_count: challenge.expected_count,
        })),
        bank.value,
      );
    } catch {
      // Invalid/empty responses are displayed per challenge; remove any stale verdict.
      session.result = null;
    }
  }

  return { sessions, start, clear, parse };
}
