import React, { useState, useRef } from "react";
import "./DetectionSection.css";

const DetectionSection = () => {
  const [image, setImage] = useState(null);
  const [result, setResult] = useState("");
  const [streaming, setStreaming] = useState(false);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  // Function to handle image upload
  const handleImageUpload = (e) => {
    console.log(e)
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = () => {
        setImage(reader.result);
        setResult(""); // Reset result
      };
      reader.readAsDataURL(file);
    }
  };

  // Function to open camera
  const openCamera = () => {
    navigator.mediaDevices
      .getUserMedia({ video: true })
      .then((stream) => {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
        setStreaming(true);
      })
      .catch(() => {
        alert("Unable to access camera. Please check your device settings.");
      });
  };

  // Function to stop camera
  const stopCamera = () => {
    const stream = videoRef.current.srcObject;
    if (stream) {
      const tracks = stream.getTracks();           //tracks means:audio and video
      tracks.forEach((track) => track.stop());
      videoRef.current.srcObject = null;
    }
    setStreaming(false);
  };

  // Function to capture image
  const captureImage = () => {
    if (!streaming) {
      alert("Camera is not active. Please start the camera first.");
      return;
    }
    const context = canvasRef.current.getContext("2d");
    const width = videoRef.current.videoWidth;
    const height = videoRef.current.videoHeight;

    // Ensure captured image matches video size
    canvasRef.current.width = width;
    canvasRef.current.height = height;

    context.drawImage(videoRef.current, 0, 0, width, height);
    const dataUrl = canvasRef.current.toDataURL("image/png");
    setImage(dataUrl);
    setResult(""); // Reset result
  };

  // Function to detect sign language
  const detectSignLanguage = () => {
    if (image) {
      // Simulate sign language detection with a placeholder result
      setTimeout(() => {
        const detectedText = "Hello"; // Example text for detected sign language
        setResult(detectedText);
        speakDetectedText(detectedText); // Speak the detected text
      }, 1000);
    } else {
      alert("Please upload or capture an image first.");
    }
  };

  // Function to speak the detected text
  const speakDetectedText = (text) => {
    if ('speechSynthesis' in window) {
      const speech = new SpeechSynthesisUtterance(text);
      speech.lang = 'en-US'; // You can change the language here
      window.speechSynthesis.speak(speech);
    } else {
      alert("Sorry, your browser does not support text-to-speech functionality.");
    }
  };

  return (
    <section>
      <div className="detection-section">
        <h2>GestureSpeak: Image/Capture Image</h2>
        <div className="detection-controls">
          {/* Image Upload */}
          <div className="upload-section">
            <label htmlFor="upload-input" className="upload-label">
              Upload an Image
            </label>
            <input
              id="upload-input"
              type="file"
              accept="image/*"
              onChange={handleImageUpload}
            />
          </div>

          {/* Open Camera */}
          <div className="camera-section">
            <div className="video-wrapper">
              {!streaming && (
                <div className="video-overlay">
                  <p>Camera feed will appear here when activated.</p>
                </div>
              )}
              <video ref={videoRef} className="camera-feed" autoPlay playsInline />
            </div>

            <div className="camera-buttons">
              <button onClick={openCamera} className="camera-btn">
                Open Camera
              </button>
              <button onClick={captureImage} className="capture-btn">
                Capture Image
              </button>
              {streaming && (
                <button onClick={stopCamera} className="stop-camera-btn">
                  Stop Camera
                </button>
              )}
            </div>
          </div>
        </div>

        {/* Display Image */}
        <div className="image-preview">
          {image && <img src={image} alt="Captured or Uploaded" />}
          <canvas ref={canvasRef} className="hidden-canvas"></canvas>
        </div>

        {/* Detect Sign Language */}
        <button onClick={detectSignLanguage} className="detect-btn">
          Detect Sign Language
        </button>

        {/* Detection Result */}
        {result && <div className="result-section">{result}</div>}
      </div>
    </section>
  );
};

export default DetectionSection;
