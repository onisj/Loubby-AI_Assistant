const typingForm = document.querySelector(".typing-form");
const chatContainer = document.querySelector(".chat-list");
const suggestions = document.querySelectorAll(".suggestion");
const toggleThemeButton = document.querySelector("#theme-toggle-button");
const deleteChatButton = document.querySelector("#delete-chat-button");
const feedbackBtn = document.querySelector("#feedback-btn");
const micBtn = document.querySelector("#mic-btn");
const videoBtn = document.querySelector("#video-btn");
const langSelect = document.querySelector("#lang-select");
const feedbackModal = document.querySelector("#feedback-modal");
const submitFeedbackBtn = document.querySelector("#submit-feedback");
const videoContainer = document.querySelector("#video-container");
const avatarVideo = document.querySelector("#avatar-video");

let userMessage = null;
let isResponseGenerating = false;
let isMicActive = false;
let isVideoActive = false;
let recognition = null;
let isFirstSession = true; // Track first session
const synth = window.speechSynthesis;
const API_URL = "http://localhost:8000/chat";

const loadDataFromLocalstorage = () => {
  const savedChats = localStorage.getItem("saved-chats");
  const isLightMode = (localStorage.getItem("themeColor") === "light_mode");
  document.body.classList.toggle("light_mode", isLightMode);
  toggleThemeButton.innerText = isLightMode ? "dark_mode" : "light_mode";
  chatContainer.innerHTML = savedChats || '';
  document.body.classList.toggle("hide-header", savedChats);
  chatContainer.scrollTo(0, chatContainer.scrollHeight);
  isFirstSession = !savedChats; // Reset on refresh if no chats
};

const createMessageElement = (content, ...classes) => {
  const div = document.createElement("div");
  div.classList.add("message", ...classes);
  div.innerHTML = content;
  return div;
};

const showTypingEffect = (text, textElement, incomingMessageDiv) => {
  const words = text.split(' ');
  let currentWordIndex = 0;
  const typingInterval = setInterval(() => {
    textElement.innerText += (currentWordIndex === 0 ? '' : ' ') + words[currentWordIndex++];
    incomingMessageDiv.querySelector(".icon").classList.add("hide");
    if (currentWordIndex === words.length) {
      clearInterval(typingInterval);
      isResponseGenerating = false;
      incomingMessageDiv.querySelector(".icon").classList.remove("hide");
      localStorage.setItem("saved-chats", chatContainer.innerHTML);
    }
    chatContainer.scrollTo(0, chatContainer.scrollHeight);
  }, 75);
};

const generateAPIResponse = async (incomingMessageDiv, query) => {
  const textElement = incomingMessageDiv.querySelector(".text");
  let chatHistory = JSON.parse(localStorage.getItem("chat-history")) || [];
  try {
    console.log("Sending query:", query);
    const response = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: query, history: chatHistory, lang: langSelect.value })
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Error fetching response");
    const apiResponse = data.answer;
    console.log("API response:", apiResponse);
    showTypingEffect(apiResponse, textElement, incomingMessageDiv);
    localStorage.setItem("chat-history", JSON.stringify(data.history));
    if (isMicActive || isVideoActive) speakResponse(apiResponse);
  } catch (error) {
    console.error("API error:", error.message);
    isResponseGenerating = false;
    textElement.innerText = "Error: Could not get response. Please try again.";
    incomingMessageDiv.classList.add("error");
  } finally {
    incomingMessageDiv.classList.remove("loading");
  }
};

const showLoadingAnimation = () => {
  const html = `<div class="message-content">
                  <img class="avatar" src="/static/images/AI.png" alt="AI avatar">
                  <p class="text"></p>
                  <div class="loading-indicator">
                    <div class="loading-bar"></div>
                    <div class="loading-bar"></div>
                    <div class="loading-bar"></div>
                  </div>
                </div>
                <span onClick="copyMessage(this)" class="icon material-symbols-rounded">content_copy</span>`;
  const incomingMessageDiv = createMessageElement(html, "incoming", "loading");
  chatContainer.appendChild(incomingMessageDiv);
  chatContainer.scrollTo(0, chatContainer.scrollHeight);
  return incomingMessageDiv;
};

const copyMessage = (copyButton) => {
  const messageText = copyButton.parentElement.querySelector(".text").innerText;
  navigator.clipboard.writeText(messageText);
  copyButton.innerText = "done";
  setTimeout(() => copyButton.innerText = "content_copy", 1000);
};

const handleOutgoingChat = () => {
  userMessage = typingForm.querySelector(".typing-input").value.trim() || userMessage;
  if (!userMessage || isResponseGenerating) return;
  isResponseGenerating = true;
  const html = `<div class="message-content">
                  <img class="avatar" src="/static/images/image.png" alt="User avatar">
                  <p class="text">${userMessage}</p>
                </div>`;
  const outgoingMessageDiv = createMessageElement(html, "outgoing");
  chatContainer.appendChild(outgoingMessageDiv);
  typingForm.reset();
  document.body.classList.add("hide-header");
  chatContainer.scrollTo(0, chatContainer.scrollHeight);
  const incomingMessageDiv = showLoadingAnimation();
  generateAPIResponse(incomingMessageDiv, userMessage);
};

const initializeRecognition = () => {
  recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
  recognition.continuous = false;
  recognition.interimResults = false;
  recognition.lang = langSelect.value;
  recognition.onresult = (event) => {
    const result = event.results[0];
    if (result.isFinal) {
      userMessage = result[0].transcript;
      console.log("Captured speech:", userMessage);
      handleOutgoingChat();
    }
  };
  recognition.onerror = (event) => {
    console.error("Speech recognition error:", event.error);
    isMicActive = false;
    isVideoActive = false;
    micBtn.classList.remove("active");
    videoBtn.classList.remove("active");
    videoContainer.style.display = "none";
  };
  recognition.onend = () => {
    if (isMicActive || isVideoActive) recognition.start();
  };
};

const speakResponse = (text) => {
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.rate = 1.0;
  const voices = synth.getVoices();
  const voice = voices.find(v => v.name.includes("Alex") || v.name.includes("Daniel")) || voices[0];
  utterance.voice = voice;
  synth.speak(utterance);
  if (isVideoActive) {
    avatarVideo.loop = true;
    avatarVideo.currentTime = 0;
    avatarVideo.play();
    utterance.onend = () => avatarVideo.pause();
  }
};

const speakGreeting = () => {
  const greeting = "Hello there, I am your Loubby Navigator. What can I help you with today?";
  const utterance = new SpeechSynthesisUtterance(greeting);
  utterance.rate = 1.0;
  const voices = synth.getVoices();
  const voice = voices.find(v => v.name.includes("Alex") || v.name.includes("Daniel")) || voices[0];
  utterance.voice = voice;
  synth.speak(utterance);
  if (isVideoActive) {
    avatarVideo.loop = false;
    avatarVideo.currentTime = 0;
    avatarVideo.play();
    utterance.onend = () => avatarVideo.pause();
  }
};

toggleThemeButton.addEventListener("click", () => {
  const isLightMode = document.body.classList.toggle("light_mode");
  localStorage.setItem("themeColor", isLightMode ? "light_mode" : "dark_mode");
  toggleThemeButton.innerText = isLightMode ? "dark_mode" : "light_mode";
});

deleteChatButton.addEventListener("click", () => {
  console.log("Delete clicked");
  if (confirm("Are you sure you want to delete all the chats?")) {
    localStorage.removeItem("saved-chats");
    localStorage.removeItem("chat-history");
    chatContainer.innerHTML = '';
    document.body.classList.remove("hide-header");
    console.log("Chat cleared");
    isFirstSession = true; // Reset for next greeting
  }
});

feedbackBtn.addEventListener("click", () => {
  feedbackModal.style.display = "flex";
});

submitFeedbackBtn.addEventListener("click", () => {
  const rating = document.getElementById("rating").value;
  const comment = document.getElementById("comment").value;
  fetch("/feedback", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ rating: parseInt(rating), comment })
  }).then(() => feedbackModal.style.display = "none");
});

micBtn.addEventListener("click", () => {
  if (!isMicActive && !isVideoActive) {
    isMicActive = true;
    micBtn.classList.add("active");
    initializeRecognition();
    if (isFirstSession) {
      speakGreeting();
      isFirstSession = false;
    }
    recognition.start();
  } else {
    isMicActive = false;
    isVideoActive = false;
    micBtn.classList.remove("active");
    videoBtn.classList.remove("active");
    videoContainer.style.display = "none";
    recognition.stop();
    synth.cancel();
  }
});

videoBtn.addEventListener("click", () => {
  if (!isVideoActive && !isMicActive) {
    isVideoActive = true;
    videoBtn.classList.add("active");
    videoContainer.style.display = "block";
    avatarVideo.load();
    initializeRecognition();
    if (isFirstSession) {
      speakGreeting();
      isFirstSession = false;
    }
    recognition.start();
  } else {
    isMicActive = false;
    isVideoActive = false;
    micBtn.classList.remove("active");
    videoBtn.classList.remove("active");
    videoContainer.style.display = "none";
    recognition.stop();
    synth.cancel();
    avatarVideo.pause();
  }
});

langSelect.addEventListener("change", () => {
  if (recognition) {
    recognition.lang = langSelect.value;
    if (isMicActive || isVideoActive) {
      recognition.stop();
      recognition.start();
    }
  }
});

suggestions.forEach(suggestion => {
  suggestion.addEventListener("click", () => {
    userMessage = suggestion.querySelector(".text").innerText;
    handleOutgoingChat();
  });
});

typingForm.addEventListener("submit", (e) => {
  e.preventDefault();
  handleOutgoingChat();
});

loadDataFromLocalstorage();