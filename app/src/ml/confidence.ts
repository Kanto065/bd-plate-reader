/**
 * Decides when the Result screen should say "Check the characters before
 * copying". Pure function, no model or UI code.
 *
 * Inputs are the class names in reading order and one score per character.
 * A read is low confidence when any of these hold:
 *   1. nothing was detected
 *   2. any character score is below minCharScore
 *   3. a character has an unknown class id ('?')
 *   4. the digit count is outside minDigits..maxDigits (a BRTA plate has two
 *      digits, a hyphen, then four digits, so six in total)
 *   5. the series letter is missing, or there is more than one
 *   6. no district word is found, or there are more than two words
 *      (a district alone, or a district followed by Metro)
 */
import { LETTERS, METRO } from './postprocess';

export interface ConfidenceOptions {
  minCharScore: number;
  minDigits: number;
  maxDigits: number;
}

export const DEFAULT_CONFIDENCE_OPTIONS: ConfidenceOptions = {
  minCharScore: 0.6,
  minDigits: 5,
  maxDigits: 6,
};

export interface ConfidenceAssessment {
  low: boolean;
  reasons: string[];
}

const DIGIT = /^[0-9]$/;

export function assessConfidence(
  tokens: readonly string[],
  scores: readonly number[],
  options: Partial<ConfidenceOptions> = {},
): ConfidenceAssessment {
  const opts = { ...DEFAULT_CONFIDENCE_OPTIONS, ...options };
  const reasons: string[] = [];

  if (tokens.length === 0) {
    return { low: true, reasons: ['no characters found'] };
  }
  if (scores.length !== tokens.length) {
    reasons.push('scores do not match characters');
  }
  if (scores.some(s => !(s >= opts.minCharScore))) {
    reasons.push('a character score is low');
  }
  if (tokens.includes('?')) {
    reasons.push('unknown character');
  }

  let digits = 0;
  let letters = 0;
  let words = 0;
  let districts = 0;
  for (const t of tokens) {
    if (DIGIT.test(t)) {
      digits += 1;
    } else if (LETTERS.has(t)) {
      letters += 1;
    } else if (t !== '?') {
      words += 1;
      if (t !== METRO) {
        districts += 1;
      }
    }
  }
  if (digits < opts.minDigits || digits > opts.maxDigits) {
    reasons.push('unexpected digit count: ' + digits);
  }
  if (letters !== 1) {
    reasons.push('expected one series letter, found ' + letters);
  }
  if (districts < 1 || words > 2) {
    reasons.push('district word missing or too many words');
  }
  return { low: reasons.length > 0, reasons };
}

export function isLowConfidence(
  tokens: readonly string[],
  scores: readonly number[],
  options: Partial<ConfidenceOptions> = {},
): boolean {
  return assessConfidence(tokens, scores, options).low;
}
