const LABEL_NAMES = {
  sports: "ورزشی",
  social: "اجتماعی",
  "culture-media": "فرهنگ و رسانه",
  economy: "اقتصادی",
  politics: "سیاسی",
};

const METRIC_NAMES = {
  accuracy: "درستی (Accuracy)",
  precision: "دقت (Precision)",
  recall: "بازیابی (Recall)",
  f1_weighted: "F1 وزنی",
  f1_macro: "F1 ماکرو",
  kappa: "کاپای کوهن",
};

const FAMILY_NAMES = {
  ml: "یادگیری ماشین",
  dl: "یادگیری عمیق",
};

const percent = new Intl.NumberFormat("fa-IR", { style: "percent", maximumFractionDigits: 1 });
const decimal = new Intl.NumberFormat("fa-IR", { minimumFractionDigits: 3, maximumFractionDigits: 3 });
const count = new Intl.NumberFormat("fa-IR", { useGrouping: false });
const date = new Intl.DateTimeFormat("fa-IR", { dateStyle: "long" });

export const labelName = (label) => LABEL_NAMES[label] ?? label;
export const metricName = (metric) => METRIC_NAMES[metric] ?? metric;
export const familyName = (family) => FAMILY_NAMES[family] ?? family;

export const formatPercent = (value) => percent.format(value);
export const formatDecimal = (value) => decimal.format(value);
export const formatCount = (value) => count.format(value);
export const formatDate = (value) => date.format(new Date(value));

// Kappa is a coefficient (it can go below zero), not a share of anything.
export const formatMetric = (metric, value) => (metric === "kappa" ? formatDecimal(value) : formatPercent(value));
