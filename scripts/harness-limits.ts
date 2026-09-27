/** Safety limits shared by the hardware measurement harnesses. */
export function validateHarnessLimits(limits: { requestCapMs: number | null | undefined; ramAbortGib: number | null | undefined }): { requestCapMs: number; ramAbortGib: number } {
  const { requestCapMs, ramAbortGib } = limits
  if (typeof requestCapMs !== 'number' || !Number.isFinite(requestCapMs) || requestCapMs <= 0 || requestCapMs > 300_000) {
    throw new Error('--request-cap-ms must be a positive finite number no greater than 300000')
  }
  if (typeof ramAbortGib !== 'number' || !Number.isFinite(ramAbortGib) || ramAbortGib < 4) {
    throw new Error('--ram-abort-gib must be a finite number at least 4')
  }
  return { requestCapMs, ramAbortGib }
}
