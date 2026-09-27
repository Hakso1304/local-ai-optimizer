// Interpretation rules v2 (docs/INTERPRETATION.md): catalog as data (rules.v2.json, 'interp-2'), decisions with a
// decision trace (verdicts.ts), and the insight panel (insights.ts). Pure and deterministic.
export { RULES, RULES_VERSION, action, cite, rule, tag, type ActionType, type Rule, type Severity } from './catalog'
export {
  coverageOf, difference, fmtDiff, speedIneligible, verdicts,
  type CandidateVerdict, type Coverage, type DecisionTrace, type Difference, type GateFailure, type GenOption, type InterpretData, type Request, type StoredRun, type Verdicts
} from './verdicts'
export { interpret, type Evidence, type Insight } from './insights'
export { label } from './label'
