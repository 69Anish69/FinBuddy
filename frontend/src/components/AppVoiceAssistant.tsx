import React, { useEffect, useMemo, useRef, useState } from "react";

export type Scenario = "current" | "best" | "worst";

export interface AppVoiceAssistantProps {
  fileUploaded?: boolean;
  lastUploadSummary?: string;
  months?: number;
  target?: "Income" | "Expense" | "Savings";
  scenario?: Scenario;

  navigate?: (route: string) => void;
  setMonths?: (m: number) => void;
  setTarget?: (t: "Income" | "Expense" | "Savings") => void;
  setScenario?: (s: Scenario) => void;

  api?: {
    forecast?: string;
    suggestions?: string;
    loanPlan?: string;
    uploadSummary?: string;
  };

  lang?: string;  // "en-IN" | "en-US"
  page?: string;  // current route
}

const HELP_EXAMPLES = [
  "Go to forecast",
  "Export forecast as PDF",
  "Give suggestions to save",
  "Pay loan faster with $50 extra",
  "Help",
];

/* ---------------- helpers ---------------- */
const getRecognition = () => {
  const w = window as any;
  return (
    w.SpeechRecognition ||
    w.webkitSpeechRecognition ||
    w.mozSpeechRecognition ||
    w.msSpeechRecognition ||
    null
  );
};

const downloadFile = async (url: string, filename: string, method: "GET" | "POST" = "POST") => {
  const res = await fetch(url, { method });
  if (!res.ok) throw new Error("download failed");
  const blob = await res.blob();
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(a.href);
};

/* ---------------- component ---------------- */
const AppVoiceAssistant: React.FC<AppVoiceAssistantProps> = ({
  fileUploaded,
  lastUploadSummary,
  months = 6,
  target = "Income",
  scenario = "current",
  navigate,
  setMonths,
  setTarget,
  setScenario,
  api,
  lang = "en-IN",
  page = "",
}) => {
  const RecognitionCtor = useMemo(getRecognition, []);
  const [supported] = useState<boolean>(!!RecognitionCtor);

  // UI state
  const [open, setOpen] = useState(false); // collapsed by default
  const [showAdvanced, setShowAdvanced] = useState(false);

  // SR state
  const [listening, setListening] = useState(false);
  const [transcript, setTranscript] = useState("");
  const [status, setStatus] = useState("");
  const [error, setError] = useState<string>("");

  // mic devices + level meter (advanced)
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [deviceId, setDeviceId] = useState<string | undefined>(undefined);
  const [micPermission, setMicPermission] = useState<"granted" | "denied" | "unknown">("unknown");
  const [level, setLevel] = useState(0);

  // refs
  const recRef = useRef<any>(null);
  const lastHeardRef = useRef<string>("");
  const streamRef = useRef<MediaStream | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const rafRef = useRef<number | null>(null);

  const speak = (text: string) => {
    try {
      const u = new SpeechSynthesisUtterance(text);
      const vs = window.speechSynthesis?.getVoices() || [];
      const en = vs.find((v) => (v.lang || "").toLowerCase().startsWith("en"));
      if (en) u.voice = en;
      window.speechSynthesis.cancel();
      window.speechSynthesis.speak(u);
    } catch {}
  };

  /* ---------- audio level meter (advanced) ---------- */
  const refreshDevices = async () => {
    try {
      const all = await navigator.mediaDevices.enumerateDevices();
      const mics = all.filter((d) => d.kind === "audioinput");
      setDevices(mics);
      if (!deviceId && mics.length) setDeviceId(mics[0].deviceId);
    } catch {}
  };
  const setupMeter = (stream: MediaStream) => {
    try { audioCtxRef.current?.close(); } catch {}
    const ctx = new (window.AudioContext || (window as any).webkitAudioContext)();
    audioCtxRef.current = ctx;
    const src = ctx.createMediaStreamSource(stream);
    const analyser = ctx.createAnalyser();
    analyser.fftSize = 1024;
    src.connect(analyser);
    analyserRef.current = analyser;

    const data = new Uint8Array(analyser.fftSize);
    const loop = () => {
      analyser.getByteTimeDomainData(data);
      let peak = 0;
      for (let i = 0; i < data.length; i++) peak = Math.max(peak, Math.abs(data[i] - 128) / 128);
      setLevel(peak);
      rafRef.current = requestAnimationFrame(loop);
    };
    rafRef.current = requestAnimationFrame(loop);
  };
  const stopMeter = () => {
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    rafRef.current = null;
    try { audioCtxRef.current?.close(); } catch {}
    audioCtxRef.current = null;
    analyserRef.current = null;
    setLevel(0);
  };

  /* ---------- SpeechRecognition wiring ---------- */
  useEffect(() => {
    if (!RecognitionCtor) return;
    const rec = new RecognitionCtor();
    rec.continuous = true;
    rec.interimResults = true;
    rec.maxAlternatives = 3;
    rec.lang = lang;

    rec.onstart = () => setStatus("Listening — speak now…");
    rec.onaudiostart = () => {};
    rec.onspeechstart = () => setStatus("Speech detected…");
    rec.onspeechend = () => setStatus("Speech ended. Processing…");
    rec.onaudioend = () => {};

    rec.onresult = (e: any) => {
      let interim = "", finalTxt = "";
      for (let i = e.resultIndex; i < e.results.length; i++) {
        const res = e.results[i];
        if (res.isFinal) finalTxt += res[0].transcript;
        else interim += res[0].transcript;
      }
      const heard = (finalTxt || interim || "").trim();
      lastHeardRef.current = heard;
      setTranscript(heard);
    };

    rec.onnomatch = () => setError("I heard audio but couldn't recognize words. Try again, closer to the mic.");

    rec.onerror = (ev: any) => {
      const code = ev?.error;
      if (code === "no-speech") {
        setError("I didn't hear anything. Speak immediately after clicking.");
        setTimeout(() => {
          try { recRef.current?.start(); setListening(true); } catch {}
        }, 500);
      } else if (code === "audio-capture") setError("No microphone available (or another app is using it).");
      else if (code === "not-allowed" || code === "service-not-allowed") setError("Microphone permission blocked.");
      else setError(code || "Voice error");
      setStatus("");
    };

    rec.onend = () => {
      setListening(false);
      stopMeter();
      const q = (lastHeardRef.current || "").trim();
      lastHeardRef.current = "";
      if (q) handleQuery(q);
    };

    recRef.current = rec;
    return () => { try { rec.stop(); } catch {}; stopMeter(); streamRef.current?.getTracks().forEach(t => t.stop()); };
  }, [RecognitionCtor, lang]);

  const ensureMic = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: deviceId ? { deviceId: { exact: deviceId } } : true,
      });
      setMicPermission("granted");
      streamRef.current = stream;
      setupMeter(stream);
      await refreshDevices();
      return true;
    } catch {
      setMicPermission("denied");
      return false;
    }
  };

  const start = async () => {
    if (!supported) { setError("SpeechRecognition not supported."); return; }
    setTranscript(""); setError(""); setStatus("");
    const ok = await ensureMic();
    if (!ok) { setError("Microphone permission denied or unavailable."); return; }
    try { recRef.current?.start(); setListening(true); } catch (e: any) { setError(e?.message || "Microphone error"); }
  };

  const stop = () => { try { recRef.current?.stop(); } catch {}; setListening(false); stopMeter(); };

  /* ---------------- intents ---------------- */
  type Intent = { name: string; match: (q: string) => boolean; run: (q: string) => Promise<void> | void; };
  const intents: Intent[] = [
    {
      name: "help",
      match: (q) => /(?:^|\b)(help|what can you do|commands|how to use)\b/.test(q),
      run: () => speak("Try: go to forecast, export forecast as PDF, give suggestions to save, or pay loan faster with fifty dollars extra."),
    },
    {
      name: "navigate",
      match: (q) => /go to|open.*(dashboard|forecast|loan|loans|upload|reports|settings|chat|goals)/.test(q),
      run: (q) => {
        if (!navigate) { speak("Navigation isn't wired yet."); return; }
        if (/forecast/.test(q)) navigate("/forecast");
        else if (/loan|loans/.test(q)) navigate("/loans");
        else if (/upload/.test(q)) navigate("/upload");
        else if (/reports/.test(q)) navigate("/reports");
        else if (/settings/.test(q)) navigate("/settings");
        else if (/chat/.test(q)) navigate("/chat");
        else if (/goals/.test(q)) navigate("/goals");
        else navigate("/");
        speak("Opening page.");
      },
    },
    {
      name: "upload-status",
      match: (q) => /did my file upload|uploaded|what did you find in my file|summary of upload|upload status/.test(q),
      run: async () => {
        if (api?.uploadSummary) {
          try {
            const res = await fetch(api.uploadSummary);
            if (res.ok) {
              const data = await res.json();
              const text = data?.summary || lastUploadSummary || "I parsed your latest file.";
              speak(text); return;
            }
          } catch {}
        }
        if (fileUploaded) speak(lastUploadSummary || "Your file is uploaded and parsed.");
        else speak("I don't see a recent file uploaded yet.");
      },
    },
    {
      name: "forecast",
      match: (q) =>
        /(what'?s|show|give).*forecast|forecast.*(income|expense|savings)/.test(q) ||
        /change to \d+ months|set \d+ months|\d+\s*(month|months|mo)\b/.test(q),
      run: async (q) => {
        let reqMonths = months;
        let reqTarget: "Income" | "Expense" | "Savings" = target;
        let reqScenario: Scenario = scenario;

        const mm = q.match(/(\d+)\s*(month|months|mo)\b/);
        if (mm) { reqMonths = Math.max(1, Math.min(36, parseInt(mm[1], 10))); setMonths?.(reqMonths); }
        if (/\bincome\b/.test(q)) { reqTarget = "Income"; setTarget?.("Income"); }
        if (/(expense|spend)/.test(q)) { reqTarget = "Expense"; setTarget?.("Expense"); }
        if (/(saving|savings)/.test(q)) { reqTarget = "Savings"; setTarget?.("Savings"); }
        if (/\bbest\b/.test(q)) { reqScenario = "best"; setScenario?.("best"); }
        if (/\bworst\b/.test(q)) { reqScenario = "worst"; setScenario?.("worst"); }
        if (/(current|baseline|default)/.test(q)) { reqScenario = "current"; setScenario?.("current"); }

        if (api?.forecast) {
          setStatus("Fetching forecast...");
          try {
            const res = await fetch(api.forecast, {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ months: reqMonths, target: reqTarget, scenario: reqScenario }),
            });
            if (res.ok) { const data = await res.json(); speak(data?.summary || "Here's the latest forecast."); return; }
          } catch {}
        }
        speak("Forecast updated. You can say 'export forecast as PDF' to download it.");
      },
    },
    {
      name: "suggestions",
      match: (q) => /(tips|advice|suggestions|how to save|improve budget|optimi[sz]e)/.test(q),
      run: async () => {
        if (api?.suggestions) {
          setStatus("Fetching suggestions...");
          try {
            const res = await fetch(api.suggestions, { method: "GET" });
            if (res.ok) {
              const data = await res.json();
              const items: string[] = data?.tips || [];
              if (items.length) { speak("Quick wins: " + items.slice(0, 3).join(". ") + "."); return; }
            }
          } catch {}
        }
        speak("Quick wins: cancel unused subscriptions, set a weekly spending cap, and automate savings on payday.");
      },
    },
    {
      name: "loan-payoff",
      match: (q) => /(loan|debt).*(pay|payoff|pay off|faster|quick|extra)/.test(q) || /(i have|i've).*(loan|debt)/.test(q),
      run: async (q) => {
        const extraMatch = q.match(/([$₹€£]?\s?)(\d+(?:\.\d+)?)\s*(dollars|bucks|pounds|rupees|inr)?\b/i);
        const extra = extraMatch ? parseFloat(extraMatch[2]) : 0;

        if (api?.loanPlan) {
          setStatus("Calculating faster payoff plan...");
          try {
            const res = await fetch(api.loanPlan, {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ extraPerMonth: isFinite(extra) ? extra : 0 }),
            });
            if (res.ok) {
              const data = await res.json();
              const mSaved = data?.monthsSaved ?? null;
              const iSaved = data?.interestSaved ?? null;
              if (mSaved != null && iSaved != null) { speak(`With an extra ${extra || 0} per month, you'll finish about ${mSaved} months sooner and save roughly ${Math.round(iSaved)} in interest.`); return; }
              if (data?.summary) { speak(data.summary); return; }
            }
          } catch {}
        }
        if (extra > 0) speak(`Adding extra ${extra} each month reduces interest and time. Prioritize highest APR first.`);
        else speak("Tell me how much extra you can pay monthly, for example: add 50 dollars extra each month.");
      },
    },
    {
      name: "page-help",
      match: (q) =>
        /(explain (this|the) (page|screen|chart)|what can i do here|help here)/i.test(q),
      run: () => {
        if (page === "/forecast") {
          speak("This page forecasts income, expenses, or savings. Use Target to switch, Months to change horizon, Scenario to test best or worst cases, and the Generate Strategy button to export a PDF.");
        } else if (page === "/goals") {
          speak("This page builds a savings plan. Pick a goal, enter price and year, then add to your plan. The bar shows progress; suggestions give quick wins.");
        } else if (page === "/loans") {
          speak("This page helps you pay loans faster. Enter balances, rates, and try adding extra per month to see months and interest saved.");
        } else if (page === "/upload") {
          speak("Upload your CSV or Excel. I’ll parse it and unlock forecast, goals, and reports.");
        } else if (page === "/reports") {
          speak("Generate summaries and export PDFs of your spending and income.");
        } else {
          speak("You can navigate, ask for forecasts, savings tips, loan payoff help, or say 'open chat' to ask anything.");
        }
      },
    },

    {
      name: "forecast-export-pdf",
      match: (q) => /(export|download).*(forecast|pdf)/.test(q) || (page === "/forecast" && /(export|download)\b/.test(q)),
      run: async () => {
        try { await downloadFile("/forecast/pdf", "forecast.pdf", "POST"); speak("Exported forecast as PDF."); }
        catch { speak("I couldn't export the forecast right now."); }
      },
    },
  ];

  // Replace handleQuery with this single version
  const handleQuery = async (raw: string) => {
    const q = (raw || "").trim().toLowerCase();

    if (!q) {
      setStatus("");
      setError("I didn't hear anything.");
      try { window.speechSynthesis.cancel(); } catch {}
      speak("I didn't hear anything. Click the mic and speak right away, or say 'help'.");
      return;
    }

    for (const intent of intents) {
      try {
        if (intent.match(q)) {
          await intent.run(q);
          setError(""); setStatus("");
          return;
        }
      } catch (e) {
        console.error("[VA] intent error:", e);
        setError("Something went wrong handling that command.");
        speak("Sorry, something went wrong while handling that.");
        return;
      }
    }

    // Default fallback (no Chat required)
    setStatus("Unknown command");
    speak("I didn't catch that. Try: " + HELP_EXAMPLES.join(", ") + ".");
    // ---- OR route to Chat (uncomment below) ----
    // try { localStorage.setItem("prefill_chat", raw); } catch {}
    // if (navigate) navigate(`/chat?q=${encodeURIComponent(raw)}`);
    // speak("Opening chat to answer that.");
  };



  /* ---------------- UI ---------------- */
  if (!supported) return null;

  // Show one-time hint for the voice assistant FAB
  const [showHint, setShowHint] = useState(() => {
    try {
      return localStorage.getItem("va_hint_dismissed") !== "1";
    } catch {
      return true;
    }
  });

  const Panel = (
    <div className="w-72 rounded-2xl border border-zinc-700 bg-zinc-900/90 backdrop-blur p-3 text-zinc-100 shadow-xl">
      {/* header */}
      <div className="flex items-center justify-between">
        <div className="text-[10px] uppercase tracking-widest opacity-70">Assistant</div>
        <button
          className="h-7 w-7 rounded-md border border-zinc-700 hover:bg-zinc-800"
          onClick={() => setOpen(false)}
          title="Close"
        >
          ×
        </button>
      </div>

      {/* status / transcript (single compact line) */}
      {(status || transcript) && (
        <div className="mt-2 text-xs">
          <span className="opacity-70">{listening ? "Listening:" : "Heard:"}</span>{" "}
          {transcript || status}
        </div>
      )}

      {/* tips + mic */}
      <div className="mt-2 flex gap-3">
        <div className="flex-1">
          <div className="text-xs font-semibold">Try saying</div>
          <ul className="mt-1 text-xs opacity-90 list-disc pl-4 space-y-0.5">
            <li>"Go to goals"</li>
            <li>"Export forecast as PDF"</li>
            <li>"Pay loan faster with $50 extra"</li>
            <li>"Help"</li>
          </ul>
        </div>
        <button
          type="button"
          onClick={listening ? stop : start}
          aria-pressed={listening}
          className={
            "relative inline-flex h-12 w-12 flex-shrink-0 items-center justify-center rounded-full border border-zinc-700 bg-zinc-950 text-zinc-100 shadow-md focus:outline-none focus:ring-2 focus:ring-indigo-500 " +
            (listening ? "animate-pulse" : "")
          }
          title={listening ? "Stop listening" : "Start listening"}
        >
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="h-6 w-6">
            <path d="M12 14a3 3 0 0 0 3-3V6a3 3 0 1 0-6 0v5a3 3 0 0 0 3 3Zm5-3a5 5 0 1 1-10 0H5a7 7 0 0 0 14 0h-2Zm-5 8a7.002 7.002 0 0 1-6.32-4H7a5 5 0 0 0 10 0h1.32A7.002 7.002 0 0 1 12 19Zm-1 3h2v-2h-2v2Z" />
          </svg>
          {listening && <span className="absolute -inset-1 rounded-full ring-2 ring-indigo-500/50" aria-hidden />}
        </button>
      </div>

      {/* errors */}
      {error && <div className="mt-2 text-[11px] text-red-300">{error}</div>}

      {/* advanced toggle */}
      <button
        className="mt-2 text-[11px] opacity-80 hover:opacity-100"
        onClick={() => setShowAdvanced((s) => !s)}
        type="button"
      >
        {showAdvanced ? "◂ Less" : "More ▸"}
      </button>

      {/* advanced section (mic picker + level + self test + manual cmd) */}
      {showAdvanced && (
        <div className="mt-2 space-y-2">
          <div>
            <label className="text-[11px] opacity-70">Microphone</label>
            <select
              className="mt-1 w-full bg-zinc-950 border border-zinc-700 rounded-md px-2 py-1 text-xs"
              value={deviceId || ""}
              onChange={(e) => setDeviceId(e.target.value || undefined)}
              onClick={refreshDevices}
            >
              {devices.length === 0 ? (
                <option value="">(click Self test to load devices)</option>
              ) : (
                devices.map((d, i) => (
                  <option key={d.deviceId || i} value={d.deviceId}>
                    {d.label || `Microphone ${i + 1}`}
                  </option>
                ))
              )}
            </select>
            <div className="mt-1 h-2 w-full bg-zinc-800 rounded-full overflow-hidden">
              <div className="h-full bg-indigo-500" style={{ width: `${Math.min(100, Math.round(level * 100))}%` }} />
            </div>
            <div className="mt-1 text-[11px] opacity-70">Support: yes • mic: {micPermission}</div>
          </div>

          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={async () => {
                const ok = await ensureMic();
                alert(`SpeechRecognition: ${supported}\nMic permission: ${ok ? "granted" : "denied"}\nDevices: ${devices.length}`);
              }}
              className="text-[11px] px-2 py-1 rounded border border-zinc-700"
            >
              Self test
            </button>
            <form
              className="flex-1 flex gap-2"
              onSubmit={(e) => {
                e.preventDefault();
                const form = e.currentTarget as HTMLFormElement;
                const input = form.elements.namedItem("voicecmd") as HTMLInputElement;
                const q = (input?.value || "").trim();
                if (q) { handleQuery(q.toLowerCase()); input.value = ""; }
              }}
            >
              <input
                name="voicecmd"
                placeholder='type: "go to goals"'
                className="flex-1 bg-zinc-950 border border-zinc-700 rounded-md px-2 py-1 text-xs"
              />
              <button type="submit" className="text-[11px] px-2 py-1 rounded border border-zinc-700">Send</button>
            </form>
          </div>
        </div>
      )}
    </div>
  );

  return (
    <>
      {/* Voice FAB + label (stacked top→bottom, with some right spacing from the FinBuddy logo) */}
      <div className="fixed bottom-6 right-20 md:right-5 z-50 flex flex-col items-end gap-3">
        {!open && (
          <div className="relative">
            <div className="flex flex-col items-center gap-1">
              <button
                type="button"
                onClick={() => setOpen(true)}
                className="h-12 w-12 rounded-full bg-indigo-600 text-white shadow-lg flex items-center justify-center hover:bg-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-400"
                aria-label="Open voice assistant"
                title="Voice assistant — click, then speak"
              >
                <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" viewBox="0 0 24 24" fill="currentColor">
                  <path d="M12 14a3 3 0 0 0 3-3V6a3 3 0 1 0-6 0v5a3 3 0 0 0 3 3Zm5-3a5 5 0 1 1-10 0H5a7 7 0 0 0 14 0h-2Zm-5 8a7.002 7.002 0 0 1-6.32-4H7a5 5 0 0 0 10 0h1.32A7.002 7.002 0 0 1 12 19Zm-1 3h2v-2h-2v2Z" />
                </svg>
              </button>
              {/* label under the icon */}
              <span className="px-1.5 py-0.5 text-[10px] rounded-md bg-zinc-900 text-zinc-100 border border-zinc-700 shadow select-none">
                Voice Assistant
              </span>
            </div>
          </div>
        )}

        {/* expanded small panel */}
        {open && Panel}
      </div>
    </>
  );
};

export default AppVoiceAssistant;
