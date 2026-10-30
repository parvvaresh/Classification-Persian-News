import { CircleAlert, Newspaper } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";

import { fetchModel, predict } from "./api.js";
import ClassifyForm from "./components/ClassifyForm.jsx";
import ModelPanel from "./components/ModelPanel.jsx";
import ResultCard from "./components/ResultCard.jsx";
import SetupNotice from "./components/SetupNotice.jsx";

export default function App() {
  const [model, setModel] = useState({ status: "loading" });
  const [modelAttempt, setModelAttempt] = useState(0);

  const [text, setText] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [pending, setPending] = useState(false);
  const inFlight = useRef(null);
  const outcome = useRef(null);

  // On short screens the outcome lands below the fold, under the form.
  useEffect(() => {
    if (!result && !error) return;
    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    outcome.current.scrollIntoView({ block: "nearest", behavior: reduceMotion ? "auto" : "smooth" });
  }, [result, error]);

  useEffect(() => {
    const controller = new AbortController();
    fetchModel(controller.signal)
      .then((info) => setModel({ status: info.ready ? "ready" : "untrained", info }))
      .catch((reason) => {
        if (reason.name !== "AbortError") setModel({ status: "unreachable", message: reason.message });
      });
    return () => controller.abort();
  }, [modelAttempt]);

  const retryModel = useCallback(() => {
    setModel({ status: "loading" });
    setModelAttempt((attempt) => attempt + 1);
  }, []);

  const cancelPrediction = () => {
    inFlight.current?.abort();
    inFlight.current = null;
    setPending(false);
  };

  const classify = async (value) => {
    cancelPrediction();
    const controller = new AbortController();
    inFlight.current = controller;
    setResult(null);
    setError(null);
    setPending(true);
    try {
      const response = await predict(value.trim(), controller.signal);
      if (inFlight.current === controller) setResult(response);
    } catch (reason) {
      if (reason.name !== "AbortError") setError(reason.message);
    } finally {
      if (inFlight.current === controller) {
        inFlight.current = null;
        setPending(false);
      }
    }
  };

  // A result always belongs to the text on screen, so editing the text drops it.
  const changeText = (value) => {
    cancelPrediction();
    setText(value);
    setResult(null);
    setError(null);
  };

  const pickExample = (value) => {
    setText(value);
    classify(value);
  };

  return (
    <div className="page">
      <header className="page-header">
        <span className="page-icon">
          <Newspaper size={24} strokeWidth={1.75} />
        </span>
        <div>
          <h1>دسته‌بندی خبر فارسی</h1>
          <p>تیتر یک خبر را وارد کنید تا مدل دستهٔ موضوعی آن را تشخیص دهد.</p>
        </div>
      </header>

      <div className="layout">
        <main className="column">
          <SetupNotice model={model} onRetry={retryModel} />
          <ClassifyForm
            text={text}
            disabled={model.status !== "ready"}
            pending={pending}
            onChange={changeText}
            onSubmit={() => classify(text)}
            onPickExample={pickExample}
          />
          <div ref={outcome} aria-live="polite">
            {error && (
              <p className="message message-error with-icon" role="alert">
                <CircleAlert size={18} strokeWidth={1.75} />
                {error}
              </p>
            )}
            {result && <ResultCard result={result} />}
          </div>
        </main>

        <ModelPanel model={model} />
      </div>
    </div>
  );
}
