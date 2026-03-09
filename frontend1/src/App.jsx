import React, { useEffect, useRef, useState } from "react";

const API_URL =
  window.location.hostname === "localhost"
    ? "http://localhost:5000"
    : "https://eduvisionai-production.up.railway.app";

// Unique user ID (in real app, this would come from login)
const USER_ID = `user_${Math.random().toString(36).substr(2, 9)}`;

export default function App() {
  const [messages, setMessages] = useState([
    { role: "system", text: "Click RECORD button to start recording your voice query." },
  ]);
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [currentStage, setCurrentStage] = useState("");
  const [lastResult, setLastResult] = useState(null);
  const [engagementState, setEngagementState] = useState("N/A");
  const [fusedScore, setFusedScore] = useState(0);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isQuizMode, setIsQuizMode] = useState(false);
  const [currentQuiz, setCurrentQuiz] = useState(null);
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  const [quizScore, setQuizScore] = useState(0);
  const [userProfile, setUserProfile] = useState(null);
  const [personalization, setPersonalization] = useState(null);
  const [difficultyLevel, setDifficultyLevel] = useState("normal");
  const [readabilityMetrics, setReadabilityMetrics] = useState(null);
  
  // Keyboard shortcut states
  const [lastKeyPressed, setLastKeyPressed] = useState("");
  const [showKeyHint, setShowKeyHint] = useState(false);
  
  // Audio feedback states
  const [audioFeedbackEnabled, setAudioFeedbackEnabled] = useState(true);
  const [lastAction, setLastAction] = useState("");

  const lastResultRef = useRef(null);
  const audioRef = useRef(null);
  const chunksRef = useRef([]);
  const streamRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const engagementIntervalRef = useRef(null);
  const webcamRef = useRef(null);
  const audioStreamRef = useRef(null);

  // Keyboard shortcut styles
  const kbdStyle = {
    backgroundColor: '#1e293b',
    border: '1px solid #475569',
    borderRadius: '3px',
    boxShadow: '0 1px 0 rgba(0,0,0,0.2)',
    color: '#94a3b8',
    display: 'inline-block',
    fontSize: '11px',
    fontWeight: 'bold',
    lineHeight: 1,
    padding: '2px 4px',
    margin: '0 2px'
  };

  // ---------------- Audio Feedback Utility ----------------
  const playAudioFeedback = (type) => {
    if (!audioFeedbackEnabled) return;
    
    try {
      const audioContext = new (window.AudioContext || window.webkitAudioContext)();
      
      // Create oscillator and gain node
      const oscillator = audioContext.createOscillator();
      const gainNode = audioContext.createGain();
      
      oscillator.connect(gainNode);
      gainNode.connect(audioContext.destination);
      
      // Set volume low to not be jarring
      gainNode.gain.setValueAtTime(0.1, audioContext.currentTime);
      
      // Different sounds for different actions
      switch(type) {
        case 'success':
          oscillator.frequency.setValueAtTime(800, audioContext.currentTime);
          oscillator.frequency.exponentialRampToValueAtTime(1200, audioContext.currentTime + 0.1);
          gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.2);
          break;
          
        case 'error':
          oscillator.frequency.setValueAtTime(400, audioContext.currentTime);
          oscillator.frequency.exponentialRampToValueAtTime(200, audioContext.currentTime + 0.2);
          gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.3);
          break;
          
        case 'start':
          oscillator.frequency.setValueAtTime(600, audioContext.currentTime);
          oscillator.frequency.exponentialRampToValueAtTime(800, audioContext.currentTime + 0.1);
          gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.15);
          break;
          
        case 'stop':
          oscillator.frequency.setValueAtTime(800, audioContext.currentTime);
          oscillator.frequency.exponentialRampToValueAtTime(400, audioContext.currentTime + 0.15);
          gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.2);
          break;
          
        case 'notification':
          oscillator.frequency.setValueAtTime(500, audioContext.currentTime);
          oscillator.frequency.setValueAtTime(600, audioContext.currentTime + 0.1);
          oscillator.frequency.setValueAtTime(500, audioContext.currentTime + 0.2);
          gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.3);
          break;
          
        case 'correct':
          oscillator.frequency.setValueAtTime(600, audioContext.currentTime);
          oscillator.frequency.setValueAtTime(800, audioContext.currentTime + 0.1);
          oscillator.frequency.setValueAtTime(1000, audioContext.currentTime + 0.2);
          gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.3);
          break;
          
        case 'incorrect':
          oscillator.frequency.setValueAtTime(400, audioContext.currentTime);
          oscillator.frequency.setValueAtTime(300, audioContext.currentTime + 0.1);
          oscillator.frequency.setValueAtTime(200, audioContext.currentTime + 0.2);
          gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.3);
          break;
          
        case 'navigate':
          oscillator.frequency.setValueAtTime(400, audioContext.currentTime);
          oscillator.frequency.setValueAtTime(450, audioContext.currentTime + 0.05);
          gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.1);
          break;
          
        case 'processing':
          // Gentle pulsating sound
          oscillator.frequency.setValueAtTime(440, audioContext.currentTime);
          oscillator.frequency.setValueAtTime(440, audioContext.currentTime + 0.5);
          gainNode.gain.setValueAtTime(0.05, audioContext.currentTime);
          gainNode.gain.setValueAtTime(0.1, audioContext.currentTime + 0.25);
          gainNode.gain.setValueAtTime(0.05, audioContext.currentTime + 0.5);
          break;
          
        case 'complete':
          oscillator.frequency.setValueAtTime(400, audioContext.currentTime);
          oscillator.frequency.setValueAtTime(500, audioContext.currentTime + 0.1);
          oscillator.frequency.setValueAtTime(600, audioContext.currentTime + 0.2);
          oscillator.frequency.setValueAtTime(700, audioContext.currentTime + 0.3);
          oscillator.frequency.setValueAtTime(800, audioContext.currentTime + 0.4);
          gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.5);
          break;
          
        default:
          oscillator.frequency.setValueAtTime(500, audioContext.currentTime);
          gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.1);
      }
      
      oscillator.start();
      oscillator.stop(audioContext.currentTime + (type === 'processing' ? 0.6 : 0.3));
      
    } catch (e) {
      console.warn("Audio feedback not supported:", e);
    }
  };

  // ---------------- Enhanced speak function with audio cues ----------------
  const speakWithFeedback = (text, feedbackType = null, callback = null) => {
    if (feedbackType) {
      playAudioFeedback(feedbackType);
      setTimeout(() => speak(text, callback), 100);
    } else {
      speak(text, callback);
    }
  };

  // ---------------- Keyboard Shortcuts Handler ----------------
  useEffect(() => {
    const handleKeyDown = (e) => {
      // Only handle Ctrl+ combinations (or Cmd on Mac)
      if (!e.ctrlKey && !e.metaKey) return;
      
      // Prevent default browser shortcuts
      e.preventDefault();
      
      const key = e.key.toLowerCase();
      
      // Show key hint
      setLastKeyPressed(`Ctrl+${key.toUpperCase()}`);
      setShowKeyHint(true);
      setTimeout(() => setShowKeyHint(false), 1500);
      
      // Ctrl + R: Start/Stop Recording Query
      if (key === 'r') {
        console.log('Ctrl+R pressed - Toggle recording');
        if (isRecording) {
          // If currently recording, stop it
          if (isQuizMode) {
            // In quiz mode, stop quiz recording
            handleQuizAnswerSubmit();
          } else {
            // In normal mode, stop recording
            stopRecording();
          }
        } else {
          // Not recording, start recording based on mode
          if (isQuizMode && currentQuiz) {
            // In quiz mode, start quiz answer recording
            startQuizRecording();
          } else {
            // In normal mode, start query recording
            startRecording();
          }
        }
      }
      
      // Ctrl + Q: Start Quiz / Stop Quiz Recording
      if (key === 'q') {
        console.log('Ctrl+Q pressed - Quiz control');
        if (isQuizMode) {
          // In quiz mode
          if (isRecording) {
            // If recording quiz answer, stop it
            handleQuizAnswerSubmit();
          } else {
            // If not recording, cancel quiz
            setCurrentQuiz(null);
            setCurrentQuestionIndex(0);
            setQuizScore(0);
            setIsQuizMode(false);
            appendMessage("system", "❌ Quiz cancelled via keyboard.");
            speakWithFeedback("Quiz cancelled.", 'stop');
          }
        } else {
          // Not in quiz mode, start quiz if content exists
          if (lastResultRef.current) {
            startQuiz();
          } else {
            appendMessage("system", "ℹ️ No content to quiz. Record a query first.");
            speakWithFeedback("Please record a query first to generate a quiz.", 'error');
          }
        }
      }
      
      // Ctrl + P: Play/Replay Last Explanation
      if (key === 'p') {
        console.log('Ctrl+P pressed - Replay explanation');
        if (lastResultRef.current?.simplified) {
          speak(lastResultRef.current.simplified);
          appendMessage("system", "🔄 Replaying explanation (Ctrl+P)");
        } else {
          appendMessage("system", "ℹ️ No explanation available to replay.");
          speakWithFeedback("No explanation available.", 'error');
        }
      }
    };

    // Add event listener
    window.addEventListener('keydown', handleKeyDown);
    
    // Cleanup
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [isRecording, isQuizMode, currentQuiz, lastResultRef.current]);

  // ---------------- Key Hint Component ----------------
  const KeyHint = () => (
    <div style={{
      position: 'fixed',
      top: '50%',
      left: '50%',
      transform: 'translate(-50%, -50%)',
      backgroundColor: 'rgba(0, 0, 0, 0.9)',
      color: '#10b981',
      padding: '20px 40px',
      borderRadius: '10px',
      fontSize: '24px',
      fontWeight: 'bold',
      zIndex: 10000,
      border: '2px solid #3b82f6',
      boxShadow: '0 0 20px rgba(59, 130, 246, 0.5)',
      transition: 'opacity 0.3s',
      opacity: showKeyHint ? 1 : 0,
      pointerEvents: 'none'
    }}>
      ⌨️ {lastKeyPressed}
    </div>
  );

  // ---------------- Audio Feedback Toggle Component ----------------
  const AudioFeedbackToggle = () => (
    <button
      onClick={() => {
        setAudioFeedbackEnabled(!audioFeedbackEnabled);
        if (!audioFeedbackEnabled) {
          playAudioFeedback('notification');
        }
        speakWithFeedback(
          audioFeedbackEnabled ? "Audio feedback disabled" : "Audio feedback enabled",
          'notification'
        );
      }}
      style={{
        padding: "8px 12px",
        backgroundColor: audioFeedbackEnabled ? "#10b981" : "#6b7280",
        color: "white",
        border: "none",
        borderRadius: "8px",
        fontSize: "14px",
        cursor: "pointer",
        display: "flex",
        alignItems: "center",
        gap: "5px"
      }}
      title={audioFeedbackEnabled ? "Disable audio feedback" : "Enable audio feedback"}
    >
      <span>🔊</span> {audioFeedbackEnabled ? "ON" : "OFF"}
    </button>
  );

  // ---------------- Load user profile on startup ----------------
  useEffect(() => {
    loadUserProfile();
  }, []);

  const loadUserProfile = async () => {
    try {
      const response = await fetch(`${API_URL}/api/get-user-profile?user_id=${USER_ID}`);
      const data = await response.json();
      if (data.success && data.profile_exists) {
        setUserProfile(data.profile);
        setDifficultyLevel(data.profile.difficulty_level || "normal");
        appendMessage("system", `📊 Welcome back! Your current learning level: ${data.profile.difficulty_level}`);
      }
    } catch (error) {
      console.log("No existing profile found, starting fresh");
    }
  };

  // ---------------- Helper: append message ----------------
  const appendMessage = (role, text) => {
    setMessages((prev) => [...prev, { role, text }]);
  };

  // ---------------- Enhanced Browser Speech Synthesis ----------------
  const speak = (text, callback = null) => {
    if (!("speechSynthesis" in window)) {
      console.warn("Browser TTS not available");
      if (callback) callback();
      return;
    }
    
    try {
      speechSynthesis.cancel();
      
      const chunkText = (textToChunk) => {
        const paragraphs = textToChunk.split(/\n\s*\n/);
        const chunks = [];
        
        for (let paragraph of paragraphs) {
          if (paragraph.trim().length === 0) continue;
          
          const sentences = paragraph.split(/(?<=[.!?])\s+/);
          let currentChunk = "";
          
          for (let sentence of sentences) {
            if ((currentChunk + sentence).length <= 200) {
              currentChunk += (currentChunk ? " " : "") + sentence;
            } else {
              if (currentChunk) chunks.push(currentChunk);
              if (sentence.length <= 200) {
                currentChunk = sentence;
              } else {
                const words = sentence.split(" ");
                currentChunk = "";
                for (let word of words) {
                  if ((currentChunk + " " + word).length <= 200) {
                    currentChunk += (currentChunk ? " " : "") + word;
                  } else {
                    if (currentChunk) chunks.push(currentChunk);
                    currentChunk = word;
                  }
                }
              }
            }
          }
          if (currentChunk) chunks.push(currentChunk);
        }
        return chunks;
      };
      
      const chunks = chunkText(text);
      let currentChunkIndex = 0;
      
      const speakNextChunk = () => {
        if (currentChunkIndex >= chunks.length) {
          if (callback) callback();
          return;
        }
        
        const utterance = new SpeechSynthesisUtterance(chunks[currentChunkIndex]);
        utterance.lang = "en-US";
        utterance.rate = 1.0;
        utterance.pitch = 1;
        utterance.volume = 1;
        
        utterance.onend = () => {
          currentChunkIndex++;
          setTimeout(speakNextChunk, 50);
        };
        
        utterance.onerror = (event) => {
          console.warn("Speech synthesis error:", event);
          currentChunkIndex++;
          setTimeout(speakNextChunk, 50);
        };
        
        try {
          speechSynthesis.speak(utterance);
        } catch (e) {
          console.warn("Failed to speak chunk:", e);
          currentChunkIndex++;
          setTimeout(speakNextChunk, 50);
        }
      };
      
      speakNextChunk();
    } catch (e) {
      console.warn("Speech synthesis failed:", e);
      if (callback) callback();
    }
  };

  // ---------------- Audio playback ----------------
  const playAudio = (url) =>
    new Promise((resolve) => {
      if (!url) return resolve();
      try {
        let fullUrl = url.startsWith("http")
          ? url
          : url.startsWith("/static/")
          ? `${API_URL}${url}`
          : `${API_URL}/static/${url}`;
        const audio = new Audio(fullUrl);
        audioRef.current = audio;
        audio.addEventListener("ended", resolve);
        audio.addEventListener("error", resolve);
        audio.play().catch(() => resolve());
      } catch {
        resolve();
      }
    });

  // ---------------- API Calls (Updated for Q-learning) ----------------
  const apiCalls = {
    transcribe: async (audioBlob) => {
      const fd = new FormData();
      fd.append("audio", audioBlob, "voice_query.webm");
      const res = await fetch(`${API_URL}/api/transcribe`, { method: "POST", body: fd });
      return await res.json();
    },
    retrieve: async (query) => {
      const res = await fetch(`${API_URL}/api/retrieve`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query }),
      });
      return await res.json();
    },
    simplify: async (content, query, difficulty) => {
      const res = await fetch(`${API_URL}/api/simplify`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ 
          content, 
          query, 
          difficulty,
          user_id: USER_ID 
        }),
      });
      return await res.json();
    },
    analyzeReadability: async (text) => {
      const res = await fetch(`${API_URL}/api/analyze-readability`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      return await res.json();
    },
    generateTTS: async (text) => {
      const res = await fetch(`${API_URL}/api/generate-tts`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      return await res.json();
    },
    generateQuiz: async (content, topic, user_id) => {
      const res = await fetch(`${API_URL}/api/generate-quiz`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ 
          content, 
          topic,
          user_id: user_id || USER_ID,
          difficulty: difficultyLevel 
        }),
      });
      return await res.json();
    },
    analyzeEngagement: async (videoBlob) => {
      const fd = new FormData();
      fd.append("video", videoBlob, "clip.webm");
      fd.append("audio", videoBlob, "clip.wav");
      fd.append("user_id", USER_ID);
      const res = await fetch(`${API_URL}/api/engagement`, { method: "POST", body: fd });
      return await res.json();
    },
    qLearningPersonalize: async (engagementData) => {
      const res = await fetch(`${API_URL}/api/qlearning-personalize`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          ...engagementData,
          user_id: USER_ID
        }),
      });
      return await res.json();
    },
    submitQuizResults: async (quizData) => {
      const res = await fetch(`${API_URL}/api/submit-quiz-results`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          ...quizData,
          user_id: USER_ID
        }),
      });
      return await res.json();
    }
  };

  // ---------------- Engagement Flow with Q-learning ----------------
  const analyzeEngagementFlow = async (manual = false) => {
    if (isAnalyzing || isRecording || isProcessing) return;
    setIsAnalyzing(true);
    
    if (manual) {
      playAudioFeedback('notification');
      appendMessage("bot", "🎥 Capturing 5 seconds of engagement data...");
      speakWithFeedback("Capturing 5 seconds of engagement data. Please look at the screen.", 'notification');
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
      const recorder = new MediaRecorder(stream);
      let chunks = [];
      
      recorder.ondataavailable = (e) => chunks.push(e.data);
      recorder.onstop = async () => {
        try {
          const blob = new Blob(chunks, { type: "video/webm" });
          const data = await apiCalls.analyzeEngagement(blob);
          
          const engagement = data.engagement_state || "Unknown";
          const videoScore = data.video_score || 0;
          const vocalScore = data.vocal_score || 0;
          const fused = data.fused_score || 0;
          
          setEngagementState(engagement);
          setFusedScore(fused);
          
          // Update personalization if available
          if (data.personalization) {
            setPersonalization(data.personalization);
            setDifficultyLevel(data.personalization.current_difficulty || difficultyLevel);
            
            if (manual) {
              playAudioFeedback('success');
              appendMessage("bot", `📊 Engagement: ${engagement} (Score: ${fused.toFixed(2)})`);
              appendMessage("bot", `🎯 Learning Level: ${data.personalization.current_difficulty}`);
              speakWithFeedback(`Your engagement level is ${engagement}. Your learning level is now ${data.personalization.current_difficulty}.`, 'success');
            }
          } else if (manual) {
            playAudioFeedback('notification');
            appendMessage("bot", `📊 Engagement Level: ${engagement} (Score: ${fused.toFixed(2)})`);
            speakWithFeedback(`Your engagement level is ${engagement}`, 'notification');
          }
          
          // Trigger Q-learning personalization
          if (fused > 0) {
            const personalizeData = {
              video_score: videoScore,
              vocal_score: vocalScore,
              fused_score: fused,
              engagement_state: engagement
            };
            
            try {
              const qlearningResult = await apiCalls.qLearningPersonalize(personalizeData);
              if (qlearningResult.success) {
                console.log("Q-learning personalization updated:", qlearningResult.personalization);
              }
            } catch (qError) {
              console.warn("Q-learning update failed:", qError);
            }
          }
          
        } catch (err) {
          console.error("Engagement API error:", err);
          playAudioFeedback('error');
        } finally {
          stream.getTracks().forEach((t) => t.stop());
          setIsAnalyzing(false);
          
          // Reload user profile
          loadUserProfile();
        }
      };
      
      recorder.start();
      setTimeout(() => recorder.stop(), 5000);
    } catch (err) {
      console.error("Engagement capture failed:", err);
      playAudioFeedback('error');
      setIsAnalyzing(false);
      if (manual) {
        appendMessage("system", "❌ Could not access camera/mic for engagement capture.");
        speakWithFeedback("Could not access camera or microphone", 'error');
      }
    }
  };

  // 🔄 Automatic engagement tracking every minute
  useEffect(() => {
    engagementIntervalRef.current = setInterval(() => {
      analyzeEngagementFlow(false);
    }, 60000);
    return () => clearInterval(engagementIntervalRef.current);
  }, []);

  // 🎥 Initialize webcam preview on right side
  useEffect(() => {
    async function initWebcam() {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: true,
          audio: false,
        });
        if (webcamRef.current) webcamRef.current.srcObject = stream;
      } catch (err) {
        console.error("Webcam access denied:", err);
      }
    }
    initWebcam();
  }, []);

  // ---------------- Cleanup recording ----------------
  const cleanupRecording = () => {
    chunksRef.current = [];
    setIsRecording(false);
    setIsProcessing(false);

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    mediaRecorderRef.current = null;
  };

  // ---------------- Start Recording ----------------
  const startRecording = async () => {
    if (isRecording || isProcessing) return;
    
    playAudioFeedback('start');
    setLastAction('start_recording');
    
    try {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
      }

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true },
      });
      streamRef.current = stream;
      const mr = new MediaRecorder(stream, { mimeType: "audio/webm;codecs=opus" });
      chunksRef.current = [];
      
      mr.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };
      
      mr.start(100);
      mediaRecorderRef.current = mr;
      setIsRecording(true);
      appendMessage("system", "🔴 Recording... Click STOP when done");
      speakWithFeedback("Recording started", 'notification');
    } catch (err) {
      console.error("startRecording error:", err);
      playAudioFeedback('error');
      appendMessage("system", "❌ Microphone permission denied.");
      speakWithFeedback("Microphone permission denied", 'error');
    }
  };

  // ---------------- Stop Recording ----------------
  const stopRecording = async () => {
    if (!isRecording || !mediaRecorderRef.current) return;

    playAudioFeedback('stop');
    setLastAction('stop_recording');
    appendMessage("system", "⏳ Processing recording...");
    speakWithFeedback("Processing your recording", 'processing');

    return new Promise((resolve) => {
      mediaRecorderRef.current.onstop = async () => {
        try {
          const blob = new Blob(chunksRef.current, { type: "audio/webm" });
          cleanupRecording();
          
          if (!blob || blob.size < 1000) {
            playAudioFeedback('error');
            appendMessage("system", "❌ Recording too short. Please try again.");
            speakWithFeedback("Recording too short, please try again", 'error');
            resolve();
            return;
          }
          
          await processRecording(blob);
          playAudioFeedback('success');
        } catch (err) {
          console.error("stopRecording error:", err);
          playAudioFeedback('error');
          appendMessage("system", "❌ Error processing recording.");
          speakWithFeedback("Error processing recording", 'error');
        } finally {
          resolve();
        }
      };

      mediaRecorderRef.current.stop();
      setIsProcessing(true);
    });
  };

  // ---------------- Process Recording with Q-learning Integration ----------------
  const processRecording = async (audioBlob) => {
    try {
      setCurrentStage("Transcribing audio…");
      const transcribeResult = await apiCalls.transcribe(audioBlob);
      const query = (transcribeResult.query || transcribeResult.text || "").trim();
      
      if (!query) {
        appendMessage("system", "❌ No speech detected. Please try again.");
        return;
      }
      
      appendMessage("user", `🗣️ ${query}`);
      speak(`Searching for ${query}`);
      appendMessage("bot", `🔍 Searching for: ${query}`);

      setCurrentStage("Retrieving content…");
      const retrieveResult = await apiCalls.retrieve(query);
      if (!retrieveResult || !retrieveResult.retrieved_content) {
        appendMessage("bot", `❌ No info found for "${query}"`);
        speak(`Sorry, I couldn't find information about ${query}`);
        return;
      }

      appendMessage("bot", `✨ Simplifying content for ${difficultyLevel} level...`);
      setCurrentStage("Simplifying content…");
      
      // Use current difficulty level from Q-learning
      const simplifyResult = await apiCalls.simplify(
        retrieveResult.retrieved_content, 
        query, 
        difficultyLevel
      );

      // Display simplification result
      const simplifiedContent = simplifyResult.simplified;
      setMessages((prev) => [
        ...prev,
        { role: "bot", text: simplifiedContent },
      ]);

      // Store readability metrics
      if (simplifyResult.readability_metrics) {
        setReadabilityMetrics(simplifyResult.readability_metrics);
        
        // Show readability info
        const readabilityMsg = `📊 Readability: FKGL ${simplifyResult.readability_metrics.final_fkgl} (Target: ${simplifyResult.readability_metrics.target_fkgl})`;
        appendMessage("system", readabilityMsg);
        
        if (!simplifyResult.readability_metrics.is_accessible) {
          appendMessage("system", "⚠️ Content may need further simplification for optimal accessibility");
        }
      }

      const newResult = { 
        query, 
        simplified: simplifiedContent, 
        tts_url: null,
        difficulty: difficultyLevel
      };
      setLastResult(newResult);
      lastResultRef.current = newResult;

      appendMessage("system", `🔊 Speaking explanation at ${difficultyLevel} level...`);
      speak(simplifiedContent, () => {
        appendMessage("system", "✅ Finished speaking explanation");
        
        setTimeout(() => {
          appendMessage("bot", "Would you like to take a quiz on this topic?");
          speak("Would you like to take a quiz on this topic?");
          
          // Show personalization info if available
          if (personalization) {
            const personalizationMsg = `🎯 Personalization: ${personalization.reason || "Adapted to your learning style"}`;
            appendMessage("system", personalizationMsg);
          }
          
          appendMessage("system", "Click START QUIZ to begin quiz or REPLAY to hear again");
        }, 1000);
      });
      
    } catch (e) {
      console.error(e);
      appendMessage("bot", `❌ Error: ${e.message || e}`);
    } finally {
      setIsProcessing(false);
      setCurrentStage("");
    }
  };

  // ---------------- Start Quiz Recording ----------------
  const startQuizRecording = async () => {
    if (isRecording || isProcessing) return;
    
    playAudioFeedback('start');
    
    try {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
      }

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true },
      });
      streamRef.current = stream;
      const mr = new MediaRecorder(stream, { mimeType: "audio/webm;codecs=opus" });
      chunksRef.current = [];
      
      mr.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };
      
      mr.start(100);
      mediaRecorderRef.current = mr;
      setIsRecording(true);
      setIsQuizMode(true);
      appendMessage("system", "🔴 Recording quiz answer... Click STOP when done");
      speakWithFeedback("Recording your answer", 'notification');
    } catch (err) {
      console.error("startQuizRecording error:", err);
      playAudioFeedback('error');
      appendMessage("system", "❌ Microphone permission denied.");
      speakWithFeedback("Microphone permission denied", 'error');
    }
  };

  // ---------------- Stop Quiz Recording ----------------
  const stopQuizRecording = async () => {
    if (!isRecording || !mediaRecorderRef.current || !isQuizMode) return;

    playAudioFeedback('stop');
    appendMessage("system", "⏳ Processing your answer...");
    speakWithFeedback("Processing your answer", 'processing');

    return new Promise((resolve) => {
      mediaRecorderRef.current.onstop = async () => {
        try {
          const blob = new Blob(chunksRef.current, { type: "audio/webm" });
          chunksRef.current = [];
          
          if (!blob || blob.size < 1000) {
            playAudioFeedback('error');
            appendMessage("system", "❌ Answer too short. Please try again.");
            speakWithFeedback("Answer too short, please try again", 'error');
            cleanupRecording();
            resolve("");
            return;
          }
          
          setIsProcessing(true);
          
          const transcribeResult = await apiCalls.transcribe(blob);
          const answer = (transcribeResult.query || transcribeResult.text || "").trim();
          
          cleanupRecording();
          resolve(answer);
        } catch (err) {
          console.error("stopQuizRecording error:", err);
          playAudioFeedback('error');
          cleanupRecording();
          resolve("");
        }
      };

      mediaRecorderRef.current.stop();
    });
  };

  // ---------------- Improved Quiz Answer Validation ----------------
  const processQuizAnswer = async (answer) => {
    if (!currentQuiz || currentQuestionIndex >= currentQuiz.questions.length) return;

    const currentQuestion = currentQuiz.questions[currentQuestionIndex];
    const correctAnswer = currentQuestion.answer || "A";
    const options = currentQuestion.options || [];
    
    appendMessage("user", `🗣️ Answer: ${answer}`);

    const userLower = answer.toLowerCase().trim();
    let isCorrect = false;
    let newScore = quizScore;

    // Extract just the letter from correct answer
    const correctLetter = correctAnswer.charAt(0).toUpperCase();
    
    // Patterns to check for
    const patterns = [
      `option ${correctLetter.toLowerCase()}`,
      `option ${correctLetter}`,
      `choice ${correctLetter.toLowerCase()}`,
      `choice ${correctLetter}`,
      `answer ${correctLetter.toLowerCase()}`,
      `answer ${correctLetter}`,
      `letter ${correctLetter.toLowerCase()}`,
      `letter ${correctLetter}`,
      `the ${correctLetter.toLowerCase()}`,
      `the ${correctLetter}`,
      `is ${correctLetter.toLowerCase()}`,
      `is ${correctLetter}`,
      `it's ${correctLetter.toLowerCase()}`,
      `it's ${correctLetter}`,
      `its ${correctLetter.toLowerCase()}`,
      `its ${correctLetter}`,
      `maybe ${correctLetter.toLowerCase()}`,
      `maybe ${correctLetter}`,
      `i think ${correctLetter.toLowerCase()}`,
      `i think ${correctLetter}`,
      `probably ${correctLetter.toLowerCase()}`,
      `probably ${correctLetter}`
    ];

    // Check if user said any of the patterns
    for (const pattern of patterns) {
      if (userLower.includes(pattern)) {
        isCorrect = true;
        break;
      }
    }

    // Also check if user just said the single letter
    if (!isCorrect && userLower === correctLetter.toLowerCase()) {
      isCorrect = true;
    }

    // Check if user said the full option text
    if (!isCorrect) {
      const correctIndex = correctLetter.charCodeAt(0) - 65;
      if (correctIndex >= 0 && correctIndex < options.length) {
        const correctOptionText = options[correctIndex].toLowerCase();
        if (userLower.includes(correctOptionText) || correctOptionText.includes(userLower)) {
          isCorrect = true;
        }
      }
    }

    if (isCorrect) {
      playAudioFeedback('correct');
      appendMessage("bot", "✅ Correct!");
      speakWithFeedback("Correct!", 'correct');
      newScore += 1;
      setQuizScore(newScore);
    } else {
      playAudioFeedback('incorrect');
      // Show correct answer
      const correctIndex = correctLetter.charCodeAt(0) - 65;
      const correctText = options[correctIndex] || "Unknown";
      const displayAnswer = `${correctLetter}: ${correctText}`;
      
      appendMessage("bot", `❌ Incorrect. Correct answer: ${displayAnswer}`);
      speakWithFeedback(`Incorrect. The correct answer is ${correctLetter}.`, 'incorrect');
    }

    // Move to next question or finish quiz
    const nextIndex = currentQuestionIndex + 1;
    if (nextIndex < currentQuiz.questions.length) {
      setCurrentQuestionIndex(nextIndex);
      setTimeout(() => askQuizQuestion(nextIndex), 2000);
    } else {
      playAudioFeedback('complete');
      const scoreMessage = `🏁 Quiz Completed! Score: ${newScore}/${currentQuiz.questions.length}`;
      appendMessage("bot", scoreMessage);
      speakWithFeedback(`Quiz completed. Your score is ${newScore} out of ${currentQuiz.questions.length}.`, 'complete');
      
      // Submit quiz results for Q-learning
      await submitQuizResults(newScore, currentQuiz.questions.length);
      
      setCurrentQuiz(null);
      setCurrentQuestionIndex(0);
      setQuizScore(0);
      setIsQuizMode(false);
    }
  };

  // ---------------- Submit Quiz Results for Q-learning ----------------
  const submitQuizResults = async (score, totalQuestions) => {
    try {
      const normalizedScore = score / totalQuestions;
      await apiCalls.submitQuizResults({
        score: normalizedScore,
        difficulty: difficultyLevel,
        num_questions: totalQuestions,
        num_correct: score
      });
      
      // Reload user profile to show updated stats
      loadUserProfile();
      
      // Trigger engagement analysis to update Q-learning
      analyzeEngagementFlow(false);
      
    } catch (error) {
      console.warn("Failed to submit quiz results:", error);
    }
  };

  // ---------------- Ask Quiz Question ----------------
  const askQuizQuestion = (index) => {
    if (!currentQuiz || index >= currentQuiz.questions.length) return;

    const question = currentQuiz.questions[index];
    const questionText = question.question || `Question ${index + 1}`;
    const options = question.options || [];

    // Display question with proper formatting
    appendMessage("bot", `Q${index + 1}: ${questionText}`);
    if (options.length > 0) {
      options.forEach((opt, idx) => {
        const letter = String.fromCharCode(65 + idx);
        appendMessage("bot", `${letter}: ${opt}`);
      });
    }

    // Speak question
    let questionToSpeak = `Question ${index + 1} of ${currentQuiz.questions.length}. `;
    questionToSpeak += `${questionText}. `;
    
    if (options.length > 0) {
      questionToSpeak += `The options are: `;
      options.forEach((opt, idx) => {
        const letter = String.fromCharCode(65 + idx);
        questionToSpeak += ` ${letter}: ${opt}.`;
      });
      questionToSpeak += ` Please say "Option" followed by the letter of your answer. For example, say "Option A".`;
    }
    
    speakWithFeedback(questionToSpeak, 'notification', () => {
      appendMessage("system", "Click RECORD ANSWER to speak your answer");
    });
  };

  // ---------------- Start Quiz with Q-learning Difficulty ----------------
  const startQuiz = async () => {
    try {
      if (!lastResultRef.current || !lastResultRef.current.simplified) {
        playAudioFeedback('error');
        appendMessage("bot", "❗ No content available to generate quiz.");
        speakWithFeedback("I don't have any content to make a quiz from.", 'error');
        return;
      }

      playAudioFeedback('notification');
      setLastAction('start_quiz');
      appendMessage("bot", `🧠 Generating ${difficultyLevel} level quiz...`);
      speakWithFeedback(`Generating ${difficultyLevel} level quiz`, 'processing');

      const quizRes = await apiCalls.generateQuiz(
        lastResultRef.current.simplified,
        lastResultRef.current.query,
        USER_ID
      );

      let questions = [];
      if (Array.isArray(quizRes)) {
        questions = quizRes;
      } else if (quizRes?.quiz && Array.isArray(quizRes.quiz)) {
        questions = quizRes.quiz;
      } else if (quizRes?.questions && Array.isArray(quizRes.questions)) {
        questions = quizRes.questions;
      } else if (quizRes?.data?.quiz && Array.isArray(quizRes.data.quiz)) {
        questions = quizRes.data.quiz;
      }

      if (questions.length === 0) {
        playAudioFeedback('error');
        appendMessage("bot", "❌ Could not generate quiz.");
        speakWithFeedback("I could not generate the quiz.", 'error');
        return;
      }

      // Ensure each question has proper structure
      const validatedQuestions = questions.map((q, i) => ({
        question: q.question || `Question ${i + 1}`,
        options: q.options || ["Option A", "Option B", "Option C", "Option D"],
        answer: q.answer || "A",
        difficulty: q.difficulty || difficultyLevel
      }));

      const quizData = {
        questions: validatedQuestions,
        topic: lastResultRef.current.query,
        difficulty: difficultyLevel
      };

      setCurrentQuiz(quizData);
      setCurrentQuestionIndex(0);
      setQuizScore(0);
      setIsQuizMode(true);

      playAudioFeedback('success');
      appendMessage("bot", `📝 Your ${difficultyLevel} level quiz has ${validatedQuestions.length} questions.`);
      
      await new Promise(resolve => {
        speakWithFeedback(`Your ${difficultyLevel} level quiz has ${validatedQuestions.length} questions. Let's begin.`, null, resolve);
      });
      
      setTimeout(() => {
        askQuizQuestion(0);
      }, 1000);
      
    } catch (err) {
      console.error("Quiz flow error", err);
      playAudioFeedback('error');
      appendMessage("bot", "❌ Error running quiz.");
      speakWithFeedback("There was an error running the quiz.", 'error');
    }
  };

  // ---------------- Handle Quiz Answer Submission ----------------
  const handleQuizAnswerSubmit = async () => {
    if (!isRecording || !isQuizMode) return;
    
    const answer = await stopQuizRecording();
    if (answer) {
      await processQuizAnswer(answer);
    } else {
      appendMessage("system", "❌ Could not process your answer. Please try again.");
      speakWithFeedback("Could not process your answer", 'error');
    }
  };

  // ---------------- View User Profile ----------------
  const viewUserProfile = () => {
    if (!userProfile) {
      playAudioFeedback('error');
      appendMessage("system", "📊 No profile data available yet. Complete a learning session first.");
      speakWithFeedback("No profile data available", 'error');
      return;
    }
    
    playAudioFeedback('notification');
    const profileMessage = `
📊 LEARNING PROFILE:
User ID: ${USER_ID}
Learning Level: ${userProfile.difficulty_level || "normal"}
Total Sessions: ${userProfile.total_sessions || 0}
Average Engagement: ${(userProfile.average_engagement * 100).toFixed(1)}%
Average Performance: ${(userProfile.average_performance * 100).toFixed(1)}%
Recent Engagement: ${(userProfile.recent_engagement * 100).toFixed(1)}%
Engagement Trend: ${userProfile.engagement_trend || "unknown"}
`;
    
    appendMessage("bot", profileMessage);
    speakWithFeedback(`Your learning profile shows you at ${userProfile.difficulty_level} level with ${userProfile.total_sessions} sessions completed.`, 'notification');
  };

  // ---------------- Analyze Readability ----------------
  const analyzeReadability = async () => {
    if (!lastResultRef.current?.simplified) {
      playAudioFeedback('error');
      appendMessage("bot", "No content available to analyze.");
      speakWithFeedback("No content available to analyze", 'error');
      return;
    }
    
    try {
      playAudioFeedback('processing');
      const result = await apiCalls.analyzeReadability(lastResultRef.current.simplified);
      if (result.success) {
        playAudioFeedback('success');
        const readabilityMsg = `
📖 READABILITY ANALYSIS:
Flesch-Kincaid Grade Level: ${result.fkgl_score}
Readability Level: ${result.readability_level}
Accessibility: ${result.metrics.is_accessible ? "✅ Good" : "⚠️ Needs improvement"}
${result.metrics.message || ""}
`;
        appendMessage("bot", readabilityMsg);
        speakWithFeedback(`The content has a readability score of ${result.fkgl_score}, which is ${result.readability_level}.`, 'success');
      }
    } catch (error) {
      console.warn("Readability analysis failed:", error);
      playAudioFeedback('error');
    }
  };

  // ---------------- Auto-scroll ----------------
  useEffect(() => {
    const el = document.querySelector(".messages");
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages]);

  return (
    <div className="app" style={{ display: "flex", height: "100vh" }}>
      {/* Keyboard Shortcut Hint Overlay */}
      <KeyHint />
      
      {/* Left: Chat Section */}
      <main className="chat-container" style={{ flex: 1, position: "relative", display: "flex", flexDirection: "column" }}>
        {/* 🧠 Enhanced Engagement Display with Q-learning Info */}
        <div
          style={{
            position: "absolute",
            top: "15px",
            right: "20px",
            background: "#0f172a",
            color: "#10b981",
            padding: "8px 16px",
            borderRadius: "12px",
            fontWeight: "bold",
            boxShadow: "0 2px 6px rgba(0,0,0,0.4)",
            zIndex: 1000,
            display: "flex",
            flexDirection: "column",
            alignItems: "flex-end",
            gap: "4px"
          }}
        >
          <div>Engagement: {engagementState} ({fusedScore.toFixed(2)})</div>
          <div style={{ fontSize: "12px", color: "#94a3b8" }}>
            Level: {difficultyLevel} | User: {USER_ID.substring(0, 8)}
          </div>
          {personalization && (
            <div style={{ fontSize: "11px", color: "#f59e0b", marginTop: "2px" }}>
              {personalization.reason?.substring(0, 40)}...
            </div>
          )}
        </div>

        {/* 📊 User Profile Summary */}
        {userProfile && (
          <div
            style={{
              position: "absolute",
              top: "15px",
              left: "20px",
              background: "rgba(15, 23, 42, 0.9)",
              color: "#94a3b8",
              padding: "8px 12px",
              borderRadius: "8px",
              fontSize: "12px",
              boxShadow: "0 2px 4px rgba(0,0,0,0.3)",
              zIndex: 1000,
              cursor: "pointer",
              border: "1px solid #1e293b"
            }}
            onClick={viewUserProfile}
            title="Click to view full profile"
          >
            📊 Profile: {userProfile.difficulty_level} | Sessions: {userProfile.total_sessions}
          </div>
        )}

        {/* Readability Metrics Display */}
        {readabilityMetrics && (
          <div
            style={{
              position: "absolute",
              top: "50px",
              left: "20px",
              background: "rgba(59, 130, 246, 0.9)",
              color: "white",
              padding: "6px 10px",
              borderRadius: "6px",
              fontSize: "11px",
              boxShadow: "0 2px 4px rgba(0,0,0,0.3)",
              zIndex: 1000,
            }}
          >
            📖 FKGL: {readabilityMetrics.final_fkgl} | Target: {readabilityMetrics.target_fkgl}
            {!readabilityMetrics.is_accessible && " ⚠️"}
          </div>
        )}

        <div
          className="messages"
          style={{ 
            padding: "20px", 
            paddingTop: "80px",
            overflowY: "auto", 
            flex: 1,
            fontFamily: "monospace"
          }}
        >
          {messages.map((m, i) => (
            <div 
              key={i} 
              className={`message ${m.role}`}
              style={{
                marginBottom: "10px",
                padding: "8px 12px",
                borderRadius: "10px",
                backgroundColor: m.role === "user" ? "#1e40af" : 
                               m.role === "bot" ? "#374151" : 
                               m.role === "system" ? "#0f172a" : "#1f2937",
                color: m.role === "system" ? "#60a5fa" : "#f3f4f6",
                maxWidth: "80%",
                alignSelf: m.role === "user" ? "flex-end" : "flex-start",
                marginLeft: m.role === "user" ? "auto" : "0",
                marginRight: m.role === "user" ? "0" : "auto",
                border: m.role === "bot" && m.text.includes("FKGL") ? "1px solid #3b82f6" : "none"
              }}
            >
              <div className="message-bubble">
                {m.text.split("\n").map((line, idx) => (
                  <p key={idx} style={{ margin: "4px 0" }}>{line}</p>
                ))}
              </div>
            </div>
          ))}
        </div>

        {/* Control Buttons with Keyboard Shortcut Labels */}
        <div style={{ 
          padding: "15px", 
          backgroundColor: "#0f172a",
          borderTop: "1px solid #1e293b",
          display: "flex",
          justifyContent: "center",
          gap: "10px",
          flexWrap: "wrap"
        }}>
          {!isQuizMode ? (
            <>
              {!isRecording ? (
                <button
                  onClick={startRecording}
                  disabled={isProcessing}
                  style={{
                    padding: "10px 20px",
                    backgroundColor: "#10b981",
                    color: "white",
                    border: "none",
                    borderRadius: "8px",
                    fontSize: "16px",
                    cursor: isProcessing ? "not-allowed" : "pointer",
                    opacity: isProcessing ? 0.6 : 1,
                    display: "flex",
                    alignItems: "center",
                    gap: "8px",
                    position: 'relative'
                  }}
                >
                  <span>🔴</span> RECORD QUERY
                  <span style={{
                    position: 'absolute',
                    bottom: '-20px',
                    left: '50%',
                    transform: 'translateX(-50%)',
                    fontSize: '10px',
                    backgroundColor: '#1e293b',
                    padding: '2px 6px',
                    borderRadius: '4px',
                    color: '#94a3b8',
                    whiteSpace: 'nowrap'
                  }}>
                    Ctrl + R
                  </span>
                </button>
              ) : (
                <button
                  onClick={stopRecording}
                  style={{
                    padding: "10px 20px",
                    backgroundColor: "#ef4444",
                    color: "white",
                    border: "none",
                    borderRadius: "8px",
                    fontSize: "16px",
                    cursor: "pointer",
                    display: "flex",
                    alignItems: "center",
                    gap: "8px",
                    position: 'relative'
                  }}
                >
                  <span>⏹️</span> STOP RECORDING
                  <span style={{
                    position: 'absolute',
                    bottom: '-20px',
                    left: '50%',
                    transform: 'translateX(-50%)',
                    fontSize: '10px',
                    backgroundColor: '#1e293b',
                    padding: '2px 6px',
                    borderRadius: '4px',
                    color: '#94a3b8',
                    whiteSpace: 'nowrap'
                  }}>
                    Ctrl + R
                  </span>
                </button>
              )}
              
              {lastResultRef.current && (
                <>
                  <button
                    onClick={startQuiz}
                    disabled={isRecording || isProcessing}
                    style={{
                      padding: "10px 20px",
                      backgroundColor: "#8b5cf6",
                      color: "white",
                      border: "none",
                      borderRadius: "8px",
                      fontSize: "16px",
                      cursor: isRecording || isProcessing ? "not-allowed" : "pointer",
                      opacity: isRecording || isProcessing ? 0.6 : 1,
                      display: "flex",
                      alignItems: "center",
                      gap: "8px",
                      position: 'relative'
                    }}
                  >
                    <span>🧠</span> START QUIZ ({difficultyLevel})
                    <span style={{
                      position: 'absolute',
                      bottom: '-20px',
                      left: '50%',
                      transform: 'translateX(-50%)',
                      fontSize: '10px',
                      backgroundColor: '#1e293b',
                      padding: '2px 6px',
                      borderRadius: '4px',
                      color: '#94a3b8',
                      whiteSpace: 'nowrap'
                    }}>
                      Ctrl + Q
                    </span>
                  </button>
                  
                  <button
                    onClick={() => {
                      if (lastResultRef.current?.simplified) {
                        speak(lastResultRef.current.simplified);
                        appendMessage("system", "🔄 Replaying explanation...");
                      }
                    }}
                    disabled={isRecording || isProcessing}
                    style={{
                      padding: "10px 20px",
                      backgroundColor: "#0ea5e9",
                      color: "white",
                      border: "none",
                      borderRadius: "8px",
                      fontSize: "16px",
                      cursor: isRecording || isProcessing ? "not-allowed" : "pointer",
                      opacity: isRecording || isProcessing ? 0.6 : 1,
                      display: "flex",
                      alignItems: "center",
                      gap: "8px",
                      position: 'relative'
                    }}
                  >
                    <span>🔄</span> REPLAY
                    <span style={{
                      position: 'absolute',
                      bottom: '-20px',
                      left: '50%',
                      transform: 'translateX(-50%)',
                      fontSize: '10px',
                      backgroundColor: '#1e293b',
                      padding: '2px 6px',
                      borderRadius: '4px',
                      color: '#94a3b8',
                      whiteSpace: 'nowrap'
                    }}>
                      Ctrl + P
                    </span>
                  </button>
                  
                  <button
                    onClick={analyzeReadability}
                    disabled={isRecording || isProcessing}
                    style={{
                      padding: "10px 20px",
                      backgroundColor: "#3b82f6",
                      color: "white",
                      border: "none",
                      borderRadius: "8px",
                      fontSize: "16px",
                      cursor: isRecording || isProcessing ? "not-allowed" : "pointer",
                      opacity: isRecording || isProcessing ? 0.6 : 1,
                      display: "flex",
                      alignItems: "center",
                      gap: "8px"
                    }}
                  >
                    <span>📖</span> READABILITY
                  </button>
                </>
              )}
              
              <button
                onClick={viewUserProfile}
                disabled={isRecording || isProcessing}
                style={{
                  padding: "10px 20px",
                  backgroundColor: "#f59e0b",
                  color: "white",
                  border: "none",
                  borderRadius: "8px",
                  fontSize: "16px",
                  cursor: isRecording || isProcessing ? "not-allowed" : "pointer",
                  opacity: isRecording || isProcessing ? 0.6 : 1,
                  display: "flex",
                  alignItems: "center",
                  gap: "8px"
                }}
              >
                <span>📊</span> PROFILE
              </button>
            </>
          ) : (
            <>
              {!isRecording ? (
                <button
                  onClick={startQuizRecording}
                  disabled={isProcessing}
                  style={{
                    padding: "10px 20px",
                    backgroundColor: "#f59e0b",
                    color: "white",
                    border: "none",
                    borderRadius: "8px",
                    fontSize: "16px",
                    cursor: isProcessing ? "not-allowed" : "pointer",
                    opacity: isProcessing ? 0.6 : 1,
                    display: "flex",
                    alignItems: "center",
                    gap: "8px"
                  }}
                >
                  <span>🎤</span> RECORD ANSWER
                </button>
              ) : (
                <button
                  onClick={handleQuizAnswerSubmit}
                  style={{
                    padding: "10px 20px",
                    backgroundColor: "#ef4444",
                    color: "white",
                    border: "none",
                    borderRadius: "8px",
                    fontSize: "16px",
                    cursor: "pointer",
                    display: "flex",
                    alignItems: "center",
                    gap: "8px"
                  }}
                >
                  <span>⏹️</span> STOP & SUBMIT
                </button>
              )}
              
              <button
                onClick={() => {
                  setCurrentQuiz(null);
                  setCurrentQuestionIndex(0);
                  setQuizScore(0);
                  setIsQuizMode(false);
                  playAudioFeedback('stop');
                  appendMessage("system", "❌ Quiz cancelled.");
                  speakWithFeedback("Quiz cancelled.", 'stop');
                }}
                disabled={isRecording || isProcessing}
                style={{
                  padding: "10px 20px",
                  backgroundColor: "#6b7280",
                  color: "white",
                  border: "none",
                  borderRadius: "8px",
                  fontSize: "16px",
                  cursor: isRecording || isProcessing ? "not-allowed" : "pointer",
                  opacity: isRecording || isProcessing ? 0.6 : 1,
                  display: "flex",
                  alignItems: "center",
                  gap: "8px"
                }}
              >
                <span>✖️</span> CANCEL QUIZ
              </button>
            </>
          )}
          
          <button
            onClick={() => analyzeEngagementFlow(true)}
            disabled={isRecording || isProcessing || isAnalyzing}
            style={{
              padding: "10px 20px",
              backgroundColor: "#6366f1",
              color: "white",
              border: "none",
              borderRadius: "8px",
              fontSize: "16px",
              cursor: isRecording || isProcessing || isAnalyzing ? "not-allowed" : "pointer",
              opacity: isRecording || isProcessing || isAnalyzing ? 0.6 : 1,
              display: "flex",
              alignItems: "center",
              gap: "8px"
            }}
          >
            <span>📊</span> CHECK ENGAGEMENT
          </button>
          
          {/* Audio Feedback Toggle Button */}
          <AudioFeedbackToggle />
        </div>

        {/* Status Display with Keyboard Shortcut Info */}
        <div className="input-area" style={{ 
          padding: "10px", 
          textAlign: "center",
          backgroundColor: "#1e293b",
          color: "#94a3b8",
          fontSize: "14px",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center"
        }}>
          <div style={{ flex: 1 }}>
            {isProcessing
              ? `⏳ ${currentStage || "Processing..."}`
              : isRecording
              ? "🔴 Recording in progress..."
              : isQuizMode && currentQuiz
              ? `🎤 Quiz Mode: Question ${currentQuestionIndex + 1} of ${currentQuiz.questions.length} | Score: ${quizScore} | Level: ${difficultyLevel}`
              : "Click buttons or use keyboard shortcuts"}
          </div>
          
          <div style={{ display: "flex", gap: "15px", alignItems: "center" }}>
            {/* Audio Feedback Indicator */}
            <div style={{ fontSize: "11px", color: audioFeedbackEnabled ? "#10b981" : "#6b7280" }}>
              🔊 {audioFeedbackEnabled ? "ON" : "OFF"}
            </div>
            
            {/* Keyboard Shortcut Indicators */}
            <div style={{ display: "flex", gap: "8px", fontSize: "11px" }}>
              <span>⌨️ <kbd style={kbdStyle}>Ctrl+R</kbd> Record</span>
              <span>⌨️ <kbd style={kbdStyle}>Ctrl+Q</kbd> Quiz</span>
              <span>⌨️ <kbd style={kbdStyle}>Ctrl+P</kbd> Replay</span>
            </div>
            
            {userProfile && (
              <div style={{ fontSize: "12px" }}>
                Sessions: {userProfile.total_sessions} | Avg. Score: {(userProfile.average_performance * 100).toFixed(0)}%
              </div>
            )}
            <div style={{ fontSize: "12px", color: "#10b981" }}>
              Q-learning: Active
            </div>
          </div>
        </div>
      </main>

      {/* Right: Live Webcam Feed & Q-learning Stats */}
      <div
        style={{
          width: "320px",
          background: "#0f172a",
          borderLeft: "2px solid #1e293b",
          display: "flex",
          flexDirection: "column",
          color: "#94a3b8",
          padding: "15px",
          overflowY: "auto"
        }}
      >
        <h3 style={{ marginBottom: "10px", color: "#10b981", textAlign: "center" }}>🎥 Live View</h3>
        <video
          ref={webcamRef}
          autoPlay
          playsInline
          muted
          style={{
            width: "100%",
            height: "auto",
            borderRadius: "10px",
            boxShadow: "0 0 10px rgba(0,0,0,0.5)",
            marginBottom: "20px"
          }}
        />
        
        {/* Q-learning Stats Panel */}
        <div style={{ 
          background: "rgba(30, 41, 59, 0.8)",
          borderRadius: "10px",
          padding: "15px",
          marginTop: "10px"
        }}>
          <h4 style={{ color: "#8b5cf6", marginBottom: "10px", textAlign: "center" }}>🤖 Q-learning Personalization</h4>
          
          <div style={{ fontSize: "12px", lineHeight: "1.6" }}>
            <div style={{ marginBottom: "8px" }}>
              <strong>Current Level:</strong> {difficultyLevel}
            </div>
            
            {personalization ? (
              <>
                <div style={{ marginBottom: "8px" }}>
                  <strong>Last Action:</strong> {personalization.action_taken || "N/A"}
                </div>
                <div style={{ marginBottom: "8px" }}>
                  <strong>Reason:</strong> {personalization.reason || "No adjustment needed"}
                </div>
                <div style={{ marginBottom: "8px" }}>
                  <strong>Engagement Score:</strong> {fusedScore.toFixed(3)}
                </div>
              </>
            ) : (
              <div style={{ marginBottom: "8px", color: "#94a3b8" }}>
                Collecting engagement data...
              </div>
            )}
            
            {userProfile && (
              <div style={{ marginTop: "15px", borderTop: "1px solid #334155", paddingTop: "10px" }}>
                <div><strong>Learning Stats:</strong></div>
                <div>Sessions: {userProfile.total_sessions}</div>
                <div>Avg Engagement: {(userProfile.average_engagement * 100).toFixed(1)}%</div>
                <div>Trend: {userProfile.engagement_trend}</div>
              </div>
            )}
            
            <div style={{ marginTop: "15px", fontSize: "11px", color: "#64748b", textAlign: "center" }}>
              User ID: {USER_ID}
            </div>
          </div>
        </div>
        
        {/* Readability Panel */}
        {readabilityMetrics && (
          <div style={{ 
            background: "rgba(59, 130, 246, 0.1)",
            borderRadius: "10px",
            padding: "15px",
            marginTop: "15px",
            border: "1px solid #3b82f6"
          }}>
            <h4 style={{ color: "#3b82f6", marginBottom: "10px", textAlign: "center" }}>📖 Readability Analysis</h4>
            
            <div style={{ fontSize: "12px", lineHeight: "1.6" }}>
              <div style={{ marginBottom: "6px" }}>
                <strong>Current FKGL:</strong> {readabilityMetrics.final_fkgl}
              </div>
              <div style={{ marginBottom: "6px" }}>
                <strong>Target FKGL:</strong> {readabilityMetrics.target_fkgl}
              </div>
              <div style={{ marginBottom: "6px" }}>
                <strong>Improvement:</strong> {readabilityMetrics.readability_improvement > 0 ? `+${readabilityMetrics.readability_improvement}` : readabilityMetrics.readability_improvement}
              </div>
              <div style={{ marginBottom: "6px", color: readabilityMetrics.is_accessible ? "#10b981" : "#f59e0b" }}>
                <strong>Accessibility:</strong> {readabilityMetrics.is_accessible ? "✅ Good" : "⚠️ Needs work"}
              </div>
              <div style={{ fontSize: "11px", color: "#94a3b8", marginTop: "8px" }}>
                {readabilityMetrics.is_accessible 
                  ? "Content is suitable for visually impaired learners"
                  : "Content may need further simplification"}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
