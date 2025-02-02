// Home.js
import React from "react";
import HeroSection from "../components/HeroSection";
import FeaturesSection from "../components/FeaturesSection";
import DetectionSection from "../components/DetectionSection"
import LiveDemoSection from "../components/LiveDemoSection"
import Dashboard from "../components/Dashboard";

const Home = () => {
  return (
    <div className="home">
    <HeroSection />
    <FeaturesSection />
    <LiveDemoSection />
    <DetectionSection/>
    <Dashboard />
    </div>
  );
};

export default Home;
