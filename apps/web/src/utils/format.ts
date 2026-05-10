export function formatLabel(value: string): string {
  return value
    .replace(/_/g, " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

export function formatNumber(value: number | null | undefined, maximumFractionDigits = 4): string {
  if (value == null || Number.isNaN(value)) {
    return "Not available";
  }
  return new Intl.NumberFormat("en-US", {
    maximumFractionDigits,
  }).format(value);
}

export function formatValue(value: unknown): string {
  if (value == null) {
    return "Not available";
  }
  if (typeof value === "number") {
    return formatNumber(value);
  }
  if (typeof value === "boolean") {
    return value ? "Yes" : "No";
  }
  if (Array.isArray(value)) {
    return value.length ? value.map((item) => formatValue(item)).join(", ") : "Not available";
  }
  if (typeof value === "object") {
    return JSON.stringify(value);
  }
  return String(value);
}
