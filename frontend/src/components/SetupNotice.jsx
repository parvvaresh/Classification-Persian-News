import { CircleAlert, Info, RotateCw } from "lucide-react";

export default function SetupNotice({ model, onRetry }) {
  if (model.status === "untrained") {
    return (
      <section className="message message-notice">
        <h2 className="with-icon">
          <Info size={18} strokeWidth={1.75} />
          هنوز مدلی آموزش داده نشده است
        </h2>
        <p>مدل را آموزش دهید و سپس سرور را دوباره اجرا کنید:</p>
        <pre dir="ltr">
          <code>{"python train.py\npython api.py"}</code>
        </pre>
      </section>
    );
  }

  if (model.status === "unreachable") {
    return (
      <section className="message message-error" role="alert">
        <p className="with-icon">
          <CircleAlert size={18} strokeWidth={1.75} />
          {model.message}
        </p>
        <p>
          سرور را با <code dir="ltr">python api.py</code> اجرا کنید و دوباره تلاش کنید.
        </p>
        <button type="button" className="button button-quiet" onClick={onRetry}>
          <RotateCw size={16} strokeWidth={1.75} />
          تلاش دوباره
        </button>
      </section>
    );
  }

  return null;
}
