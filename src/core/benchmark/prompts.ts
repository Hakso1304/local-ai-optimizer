// Deterministic ladder prompts: same ctx → same text on every machine/model.
import { generateFiller } from '../quality'

/** Share of the context the prompt fills. generateFiller assumes ≈4 chars/token; the slack also leaves room for
 *  tokenizer differences and the decoded tokens. */
export const LADDER_FILL = 0.75
export const LADDER_PREDICT = 128

/** Prefill-heavy prompt sized for a server started with -c ctx. Seed = ctx, so steps differ but are reproducible. */
export function ladderPrompt(ctx: number, fill = LADDER_FILL): string {
  const body = generateFiller(Math.floor(ctx * fill), ctx).join(' ')
  return `${body}\n\nContinue the story in the same style:\n`
}
