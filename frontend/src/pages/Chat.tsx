import React, { useEffect, useRef, useState } from "react";
import { useLocation } from "react-router-dom";

type Msg = { role: "user" | "assistant"; content: string };

const ASK_API = "http://localhost:8000/assistant/ask"; // ← change if different

const Chat: React.FC = () => {
  const { search } = useLocation();
  const [messages, setMessages] = useState<Msg[]>([
    { role: "assistant", content: "Hi! Ask me anything about your forecasts, goals, loans, or uploads." },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const endRef = useRef<HTMLDivElement>(null);

  // Auto-scroll
  useEffect(() => { endRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages]);

  // Pick up a question passed by voice assistant (?q=...) or localStorage
  useEffect(() => {
    const sp = new URLSearchParams(search);
    const q = sp.get("q") || localStorage.getItem("prefill_chat") || "";
    if (q) {
      localStorage.removeItem("prefill_chat");
      setInput(q);
      handleSend(q);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleSend = async (text?: string) => {
    const content = (text ?? input).trim();
    if (!content || loading) return;

    setMessages((m) => [...m, { role: "user", content }]);
    setInput("");
    setLoading(true);

    try {
      const res = await fetch(ASK_API, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: content }),
      });

      if (res.ok) {
        const data = await res.json();
        const answer = data.answer || "Here’s what I found.";
        setMessages((m) => [...m, { role: "assistant", content: answer }]);
      } else {
        throw new Error(String(res.status));
      }
    } catch {
      // graceful fallback if you don’t have the endpoint yet
      setMessages((m) => [
        ...m,
        {
          role: "assistant",
          content:
            "I couldn’t reach the answer service. If your backend exposes an endpoint (e.g., POST /assistant/ask → {answer}), update ASK_API in Chat.tsx.",
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-subtle">
      <div className="container mx-auto max-w-3xl px-4 py-8">
        <h1 className="text-2xl font-bold mb-4">Ask FinBuddy</h1>

        <div className="rounded-lg border bg-card shadow-sm p-4 h-[70vh] overflow-y-auto">
          {messages.map((m, i) => (
            <div key={i} className={`mb-3 flex ${m.role === "user" ? "justify-end" : "justify-start"}`}>
              <div
                className={`max-w-[80%] rounded-2xl px-3 py-2 text-sm ${
                  m.role === "user" ? "bg-indigo-600 text-white" : "bg-zinc-100 text-zinc-900"
                }`}
              >
                {m.content}
              </div>
            </div>
          ))}
          <div ref={endRef} />
        </div>

        <form
          className="mt-4 flex gap-2"
          onSubmit={(e) => {
            e.preventDefault();
            handleSend();
          }}
        >
          <input
            className="flex-1 rounded-md border px-3 py-2"
            placeholder="Type your question…"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            disabled={loading}
          />
          <button
            type="submit"
            className="rounded-md bg-indigo-600 text-white px-4 py-2 disabled:opacity-50"
            disabled={loading || !input.trim()}
          >
            {loading ? "Sending…" : "Send"}
          </button>
        </form>
      </div>
    </div>
  );
};

export default Chat;
