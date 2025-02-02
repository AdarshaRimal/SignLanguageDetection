/* eslint-disable no-unused-vars */
import React, { useState, useRef, useEffect } from 'react';
import './LiveDemoSection.css';

const LiveDemoSection = () => {
  const videoRef = useRef(null); // Reference for the video element
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [detectedText, setDetectedText] = useState("Start performing gestures to detect text...");
  const [errorMessage, setErrorMessage] = useState("");

  // Start demo - access the user's camera
  const startDemo = async () => {
    try {
      setErrorMessage(""); // Clear any previous error messages
      setIsCameraActive(true);

      if (!videoRef.current) {
        console.error('Video element not found.');
        setErrorMessage("Unable to access video feed.");
        return;
      }

      // Request camera access
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      videoRef.current.srcObject = stream; // Set the video element source to the camera stream
    } catch (err) {
      console.error("Error accessing the camera: ", err);
      setErrorMessage("Camera access is required to run the demo.");
      alert("Camera access is required to run the demo.");
    }
  };

  // Reset demo - stop the camera stream
  const resetDemo = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const tracks = videoRef.current.srcObject.getTracks();
      tracks.forEach((track) => track.stop()); // Stop all video tracks
      videoRef.current.srcObject = null;
    }
    setIsCameraActive(false);
    setDetectedText("Start performing gestures to detect text...");
  };

  // Simulate gesture detection
  useEffect(() => {
    let interval;
    if (isCameraActive) {
      interval = setInterval(() => {
        // Simulate detecting gestures and updating text
        const gestures = ["Hello", "Thank you", "Please", "Goodbye"];
        const randomGesture = gestures[Math.floor(Math.random() * gestures.length)];
        setDetectedText(randomGesture);
      }, 5000); // Update every 5 seconds
    }

    return () => clearInterval(interval); // Clean up interval on unmount or camera stop
  }, [isCameraActive]);

  // Function to speak the detected text
  const speakText = () => {
    const speech = new SpeechSynthesisUtterance(detectedText);
    window.speechSynthesis.speak(speech);
  };

  return (
    <section className="live-demo">
      <h2>GestureSpeak: Live Demo</h2>
      <p>Perform sign language gestures to see them detected in real-time.</p>

      {/* Webcam Input (Video Feed) */}
      <div className="demo-area">
        {!isCameraActive && (
          <div className="demo-placeholder">
            <p>{errorMessage || "Camera feed will appear here."}</p>
          </div>
        )}
        <video ref={videoRef} autoPlay playsInline className="video-feed" />
      </div>

      {/* Output Panel */}
      <div className="output-panel">
        <h3>Detected Gesture:</h3>
        <p>{detectedText}</p>
        <button onClick={speakText}>Speak Text</button>
      </div>

      {/* Demo Control Buttons */}
      <div className="demo-buttons">
        <button onClick={startDemo}>Start Demo</button>
        <button onClick={resetDemo}>Stop Demo</button>
      </div>
    </section>
  );
};

export default LiveDemoSection;
