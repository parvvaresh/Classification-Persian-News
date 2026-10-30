import { CircleOff, Cpu, Gauge, LoaderCircle, Tags } from "lucide-react";

import { familyName, formatDate, formatMetric, labelName, metricName } from "../format.js";
import { labelIcon } from "../icons.js";

function ModelTitle() {
  return (
    <h2 className="section-title">
      <Cpu size={16} strokeWidth={1.75} />
      مدل
    </h2>
  );
}

export default function ModelPanel({ model }) {
  if (model.status !== "ready") {
    return (
      <aside className="card panel" aria-busy={model.status === "loading"}>
        <ModelTitle />
        {model.status === "loading" ? (
          <p className="hint with-icon">
            <LoaderCircle className="spin" size={16} strokeWidth={1.75} />
            در حال بارگذاری…
          </p>
        ) : (
          <p className="hint with-icon">
            <CircleOff size={16} strokeWidth={1.75} />
            مدلی بارگذاری نشده است.
          </p>
        )}
      </aside>
    );
  }

  const { name, family, labels, test_metrics: testMetrics, created_at: createdAt } = model.info;

  return (
    <aside className="card panel">
      <ModelTitle />
      <p className="model-name" dir="ltr">
        {name}
      </p>
      <p className="hint">
        {familyName(family)} · {formatDate(createdAt)}
      </p>

      <h2 className="section-title">
        <Gauge size={16} strokeWidth={1.75} />
        نتیجه روی دادهٔ آزمون
      </h2>
      <dl className="metrics">
        {Object.entries(testMetrics).map(([metric, value]) => (
          <div key={metric}>
            <dt>{metricName(metric)}</dt>
            <dd>{formatMetric(metric, value)}</dd>
          </div>
        ))}
      </dl>

      <h2 className="section-title">
        <Tags size={16} strokeWidth={1.75} />
        دسته‌ها
      </h2>
      <ul className="labels">
        {labels.map((label) => {
          const Icon = labelIcon(label);
          return (
            <li key={label} className="with-icon">
              <Icon size={14} strokeWidth={1.75} />
              {labelName(label)}
            </li>
          );
        })}
      </ul>
    </aside>
  );
}
