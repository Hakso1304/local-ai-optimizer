// Deterministic ladder prompts: same ctx → same text on every machine/model.
import { generateFiller } from '../quality'

/** Bump when ladderPrompt output changes for any ctx. */
export const PROMPT_VERSION = 'ladder-2'
/** Share of the context the prompt fills, in the loaded model's tokens: the runner resizes the filler with the
 *  runtime tokenizer (ladder-2). ladder-1 relied on ≈4 chars/token and measured ≈ 0.56·ctx on Llama. */
export const LADDER_FILL = 0.75
/** Accept a tokenized size within this share of the target; at most this many resize rounds. */
export const LADDER_FILL_TOLERANCE = 0.03
export const LADDER_SIZE_ROUNDS = 3
export const LADDER_PREDICT = 128

/** Prefill-heavy prompt sized for a server started with -c ctx. Seed = ctx, so steps differ but are reproducible. */
export function ladderPrompt(ctx: number, fill = LADDER_FILL): string {
  const body = generateFiller(Math.floor(ctx * fill), ctx).join(' ')
  return `${body}\n\nContinue the story in the same style:\n`
}
