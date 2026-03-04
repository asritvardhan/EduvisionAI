import React, { useEffect, useRef, useState } from "react";

const API =
  window.location.hostname === "localhost"
    ? "http://localhost:5000"
    : "";

export default function App() {
  /* ---------------- STATE ---------------- */
  const [messages, setMessages] = useState([
    { role: "system", text: "Press SPACE to start or stop voice recording." }
  ]);

  const [recording, setRecording] = useState(false);
  const [processing, setProcessing] = useState(false);

  const [simplifiedText, setSimplifiedText] = useState("");
  const [ttsUrl, setTtsUrl] = useState(null);

  const [engagement, setEngagement] = useState("N/A");
  const [quiz, setQuiz] = useState([]);
  const [quizIndex, setQuizIndex] = useState(0);
  const [score, setScore] = useState(0);
  const [quizMode, setQuizMode] = useState(false);

  /* ---------------- REFS ---------------- */
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const engagementTimer = useRef(null);

  /* ---------------- CAMERA ---------------- */
  useEffect(() => {
    navigator.mediaDevices.getUserMedia({ video: true, audio: true })
      .then(stream => {
        streamRef.current = stream;
        videoRef.current.srcObject = stream;
      });

    engagementTimer.current = setInterval(runEngagementCheck, 30000);

    return () => clearInterval(engagementTimer.current);
  }, []);

  /* ---------------- KEYBOARD ---------------- */
  useEffect(() => {
    const onKey = e => {
      if (e.code === "Space") {
        e.preventDefault();
        recording ? stopRecording() : startRecording();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [recording]);

  /* ---------------- RECORDING ---------------- */
  const startRecording = () => {
    audioChunksRef.current = [];
    const recorder = new MediaRecorder(streamRef.current);
    recorder.ondataavailable = e => audioChunksRef.current.push(e.data);
    recorder.start();
    mediaRecorderRef.current = recorder;
    setRecording(true);
  };

  const stopRecording = async () => {
    mediaRecorderRef.current.stop();
    setRecording(false);
    setProcessing(true);

    const audioBlob = new Blob(audioChunksRef.current, { type: "audio/webm" });
    const fd = new FormData();
    fd.append("audio", audioBlob);

    const res = await fetch(`${API}/api/transcribe`, { method: "POST", body: fd });
    const data = await res.json();

    addMsg("user", data.query);
    handleQuery(data.query);
  };

  /* ---------------- QUERY PIPELINE ---------------- */
  const handleQuery = async query => {
    const r1 = await fetch(`${API}/api/retrieve`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query })
    }).then(r => r.json());

    const r2 = await fetch(`${API}/api/simplify`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        content: r1.retrieved_content,
        query
      })
    }).then(r => r.json());

    setSimplifiedText(r2.simplified);
    addMsg("assistant", r2.simplified);

    const r3 = await fetch(`${API}/api/generate-tts`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: r2.simplified })
    }).then(r => r.json());

    playTTS(r3.tts_url);
  };

  /* ---------------- TTS ---------------- */
  const playTTS = url => {
    const audio = new Audio(`${API}${url}`);
    audio.onended = askReplayOrQuiz;
    audio.play();
  };

  /* ---------------- DECISION ---------------- */
  const askReplayOrQuiz = () => {
    speak("Say replay to listen again, or quiz to start questions.");
  };

  const speak = async text => {
    const r = await fetch(`${API}/api/generate-tts`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text })
    }).then(r => r.json());
    new Audio(`${API}${r.tts_url}`).play();
  };

  /* ---------------- QUIZ ---------------- */
  const startQuiz = async () => {
    const r = await fetch(`${API}/api/generate-quiz`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        content: simplifiedText,
        topic: "Topic"
      })
    }).then(r => r.json());

    setQuiz(r.quiz);
    setQuizIndex(0);
    setScore(0);
    setQuizMode(true);
    askQuestion(r.quiz[0]);
  };

  const askQuestion = q => {
    speak(`${q.question}. Options are ${q.options.join(", ")}`);
  };

  /* ---------------- ENGAGEMENT ---------------- */
  const runEngagementCheck = async () => {
    const fd = new FormData();
    fd.append("video", new Blob([], { type: "video/mp4" }));
    fd.append("audio", new Blob([], { type: "audio/wav" }));

    const r = await fetch(`${API}/api/engagement`, {
      method: "POST",
      body: fd
    }).then(r => r.json());

    setEngagement(r.engagement_state);
  };

  /* ---------------- UTILS ---------------- */
  const addMsg = (role, text) =>
    setMessages(m => [...m, { role, text }]);

  /* ---------------- UI ---------------- */
  return (
    <div style={{ display: "flex", height: "100vh" }}>
      {/* CHAT */}
      <div style={{ flex: 2, padding: 20, overflowY: "auto" }}>
        {messages.map((m, i) => (
          <div key={i} style={{ marginBottom: 12 }}>
            <b>{m.role}:</b> {m.text}
          </div>
        ))}
        {recording && <p>🎤 Recording...</p>}
        {processing && <p>⏳ Processing...</p>}
      </div>

      {/* CAMERA */}
      <div style={{ flex: 1, padding: 10 }}>
        <video ref={videoRef} autoPlay muted style={{ width: "100%" }} />
        <h3>Engagement: {engagement}</h3>
      </div>
    </div>
  );
}
