import { ChartBar, Target } from "lucide-react";

import { formatDecimal, formatPercent, labelName } from "../format.js";
import { labelIcon } from "../icons.js";

export default function ResultCard({ result }) {
  const format = result.is_probability ? formatPercent : formatDecimal;
  const TopIcon = labelIcon(result.label);

  return (
    <section className="card result">
      <p className="field-label">
        <Target size={16} strokeWidth={1.75} />
        دستهٔ تشخیص‌داده‌شده
      </p>
      <p className="result-label">
        <TopIcon size={40} strokeWidth={1.5} />
        {labelName(result.label)}
      </p>
      {result.is_probability ? (
        <p className="hint">احتمال {formatPercent(result.score)}</p>
      ) : (
        <p className="hint">این مدل احتمال واقعی نمی‌دهد؛ امتیازها فقط بین دسته‌ها قابل مقایسه‌اند.</p>
      )}

      <h2 className="section-title">
        <ChartBar size={16} strokeWidth={1.75} />
        {result.is_probability ? "احتمال هر دسته" : "امتیاز هر دسته"}
      </h2>
      <ol className="scores">
        {result.scores.map(({ label, score }) => {
          const Icon = labelIcon(label);
          return (
            <li key={label} className={label === result.label ? "score score-top" : "score"}>
              <span className="score-name with-icon">
                <Icon size={16} strokeWidth={1.75} />
                {labelName(label)}
              </span>
              <span className="score-track" aria-hidden="true">
                <span className="score-bar" style={{ width: `${Math.min(score, 1) * 100}%` }} />
              </span>
              <span className="score-value">{format(score)}</span>
            </li>
          );
        })}
      </ol>
    </section>
  );
}
