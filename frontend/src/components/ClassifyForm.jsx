import { Lightbulb, LoaderCircle, PenLine, ScanSearch } from "lucide-react";

import { MAX_TEXT_LENGTH } from "../api.js";
import { formatCount } from "../format.js";

const EXAMPLES = [
  "پرسپولیس با برتری مقابل استقلال قهرمان شد",
  "بانک مرکزی نرخ سود تسهیلات بانکی را اعلام کرد",
  "وزیر امور خارجه با همتای خود دربارهٔ روابط دو کشور گفت‌وگو کرد",
  "جشنواره فیلم فجر با نمایش فیلم‌های تازه آغاز شد",
  "طرح جدید پیشگیری از آسیب‌های اجتماعی در مدارس اجرا می‌شود",
];

export default function ClassifyForm({ text, disabled, pending, onChange, onSubmit, onPickExample }) {
  const canSubmit = !disabled && !pending && text.trim().length > 0;

  const submit = (event) => {
    event.preventDefault();
    if (canSubmit) onSubmit();
  };

  const submitOnShortcut = (event) => {
    if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) submit(event);
  };

  return (
    <form className="card" onSubmit={submit}>
      <label className="field-label" htmlFor="news-text">
        <PenLine size={16} strokeWidth={1.75} />
        متن خبر
      </label>
      <textarea
        id="news-text"
        rows={5}
        dir="auto"
        maxLength={MAX_TEXT_LENGTH}
        placeholder="برای نمونه: پرسپولیس با برتری مقابل استقلال قهرمان شد"
        value={text}
        disabled={disabled}
        onChange={(event) => onChange(event.target.value)}
        onKeyDown={submitOnShortcut}
      />

      <div className="form-actions">
        <button type="submit" className="button" disabled={!canSubmit}>
          {pending ? (
            <LoaderCircle className="spin" size={18} strokeWidth={1.75} />
          ) : (
            <ScanSearch size={18} strokeWidth={1.75} />
          )}
          {pending ? "در حال تحلیل…" : "دسته‌بندی کن"}
        </button>
        <span className="hint">
          {formatCount(text.length)} از {formatCount(MAX_TEXT_LENGTH)} نویسه
        </span>
      </div>

      <div className="examples">
        <span className="hint with-icon" id="examples-label">
          <Lightbulb size={16} strokeWidth={1.75} />
          نمونه‌ها
        </span>
        <ul aria-labelledby="examples-label">
          {EXAMPLES.map((example) => (
            <li key={example}>
              <button type="button" className="chip" disabled={disabled} onClick={() => onPickExample(example)}>
                {example}
              </button>
            </li>
          ))}
        </ul>
      </div>
    </form>
  );
}
