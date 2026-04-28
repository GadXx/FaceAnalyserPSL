import type { Locale } from "../utils/i18n";

export interface LocalizedHelpEntry {
  label?: string;
  short: string;
  extended?: string;
  interpretation?: string;
  warning?: string;
}

export function resolveHelpKey(key: string | undefined | null): string | null;
export function getHelpEntry(key: string | undefined | null, locale: Locale): LocalizedHelpEntry | null;
export function getHelpLabel(
  key: string | undefined | null,
  locale: Locale,
  fallbackLabel?: string | undefined | null,
): string;
