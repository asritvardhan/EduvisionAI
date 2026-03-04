import React from "react";
import { FaVolumeUp } from "react-icons/fa";

const ChatMessage = ({ msg }) => {
  const playAudio = () => {
    if (msg.audio) {
      const audio = new Audio(msg.audio);
      audio.play();
    }
  };

  return (
    <div className={`chat-message ${msg.sender}`}>
      <div className="message-bubble">
        {msg.text}
        {msg.sender === "bot" && msg.audio && (
          <button className="replay-btn" onClick={playAudio}>
            <FaVolumeUp />
          </button>
        )}
      </div>
    </div>
  );
};

export default ChatMessage;
