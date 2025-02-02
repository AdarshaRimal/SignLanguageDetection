import React from "react";
import "./TryDemo.css";
import LiveDemoSection from "../components/LiveDemoSection";
import DetectionSection from "../components/DetectionSection";

const TryDemo = () => {
  return (
    <>
      <div className="try-demo-page ">
      <header className="try-demo-header">
        <h1>Try the Demo</h1>
        <p>
          Experience the power of GestureSpeak by trying out our interactive demo. In this feature, you can perform sign language gestures, and watch as they are instantly translated into text. Whether you're new to sign language or looking for a quick and fun way to see how our platform works, this demo is the perfect way to get started.
        </p>
      </header>

      <section className="demo-description">
        <h2>How It Works:</h2>
        <ol>
          <li><strong>Start the Demo</strong>: Click on the "Start Demo" button to begin.</li>
          <li><strong>Perform a Gesture</strong>: Use your hand to mimic the sign language gestures that the platform can recognize.</li>
          <li><strong>See the Result</strong>: Watch as your gesture is instantly converted into text on the screen.</li>
        </ol>

        <h2>Why Try Demo?</h2>
        <ul>
          <li><strong>Learn in Action</strong>: Get a real-time view of how gesture recognition works.</li>
          <li><strong>Instant Feedback</strong>: See the translation of your gestures immediately.</li>
          <li><strong>Accessible</strong>: Designed for users of all levels to experience gesture recognition in action.</li>
        </ul>
      </section>
      </div>
      <LiveDemoSection />
      <DetectionSection />
    </>
  )
}
export default TryDemo;